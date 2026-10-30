/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/

// Scratch-disk caching of particle images.
//
// Particles are copied to a local scratch disk as raw bytes: the MRC/MRCS
// payload is never parsed or converted, so the on-disk representation
// (float16 included) is preserved and the reader does the conversion when it
// loads the image.

#include <map>
#include <set>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <omp.h>
#include "src/exp_model.h"
#include <sys/stat.h>
#include <sys/statvfs.h>
using namespace gravis;

namespace
{
	// Minimal view on the fixed 1024-byte MRC header, used by the raw-byte
	// scratch copy. Only the fields needed to locate a slice in the source
	// stack and to keep the data representation intact are read.
	struct MrcHeaderInfo
	{
		int32_t nx = 0, ny = 0, nz = 0, mode = -1, nsymbt = 0;
		bool swapped = false;
	};

	// Returns false for anything that should not be raw-copied (non-MRC files,
	// unreadable headers or unsupported modes).
	bool readMrcHeader(const FileName &fn, MrcHeaderInfo &h)
	{
		FILE *fin = fopen(fn.c_str(), "rb");
		if (!fin)
			return false;

		char buf[1024];
		bool got_header = (fread(buf, 1, sizeof(buf), fin) == sizeof(buf));
		fclose(fin);
		if (!got_header)
			return false;

		memcpy(&h.nx,     buf +  0, sizeof(int32_t));
		memcpy(&h.ny,     buf +  4, sizeof(int32_t));
		memcpy(&h.nz,     buf +  8, sizeof(int32_t));
		memcpy(&h.mode,   buf + 12, sizeof(int32_t));
		memcpy(&h.nsymbt, buf + 92, sizeof(int32_t));

		// Same test the MRC reader uses to spot a byte-swapped header
		// (see parseMRCHeader): swap the integer fields before validating.
		// The raw copy leaves the bytes untouched, so the file keeps its
		// byte order and Image::read will still convert it correctly.
		if (std::abs(h.mode) > SWAPTRIG || std::abs(h.nx) > SWAPTRIG)
		{
			h.swapped = true;
			int32_t *fields[] = {&h.nx, &h.ny, &h.nz, &h.mode, &h.nsymbt};
			for (int32_t *f : fields)
			{
				char *p = (char *) f;
				std::swap(p[0], p[3]);
				std::swap(p[1], p[2]);
			}
		}

		// Implausible dimensions mean a corrupt header.
		if (h.nx <= 0 || h.ny <= 0 || h.nz <= 0 || h.nsymbt < 0)
			return false;

		switch (h.mode)
		{
		case 0:   // signed char
		case 1:   // signed short
		case 2:   // float32
		case 6:   // unsigned short
		case 12:  // float16
		case 101: // 4-bit (SerialEM)
			return true;
		default:
			return false;
		}
	}

	// Bytes occupied by one image (one z-slice) of an MRC stack.
	bool mrcBytesPerImage(const MrcHeaderInfo &h, size_t &bytes_per_image)
	{
		const size_t n_pixels = (size_t) h.nx * (size_t) h.ny;

		size_t bytes_per_pixel = 0;
		switch (h.mode)
		{
		case 0:  bytes_per_pixel = 1; break;
		case 1:
		case 6:
		case 12: bytes_per_pixel = 2; break;
		case 2:  bytes_per_pixel = 4; break;
		case 101:
			// Two 4-bit pixels are packed into one byte. parseMRCHeader()
			// rejects this mode when both dimensions are odd, so do the same.
			if (h.nx % 2 != 0 && h.ny % 2 != 0)
				return false;
			bytes_per_image = n_pixels / 2;
			return true;
		default:
			return false;
		}

		bytes_per_image = n_pixels * bytes_per_pixel;
		return true;
	}

	// Write a 32-bit header field in the byte order of the source header, so
	// patching a foreign-endian stack keeps producing a readable file.
	void storeInt32(char *p, int32_t value, bool swapped)
	{
		char *v = (char *) &value;
		if (swapped)
		{
			p[0] = v[3];
			p[1] = v[2];
			p[2] = v[1];
			p[3] = v[0];
		}
		else
		{
			memcpy(p, v, 4);
		}
	}

	// File length in bytes, or 0 when the file cannot be stat'ed.
	size_t fileSize(const FileName &fn)
	{
		struct stat st;
		if (stat(fn.c_str(), &st) != 0)
			return 0;
		return (size_t) st.st_size;
	}

	// Stream nbytes from fin to fout through a reusable chunk buffer.
	// Whole subtomogram files can be hundreds of MB, so they are never
	// buffered in full.
	bool copyBytes(FILE *fin, FILE *fout, size_t nbytes, std::vector<char> &chunk)
	{
		const size_t chunk_size = 8u << 20; // 8 MB

		while (nbytes > 0)
		{
			const size_t n = std::min(nbytes, chunk_size);
			if (chunk.size() < n)
				chunk.resize(n);

			if (fread(chunk.data(), 1, n, fin) != n)
				return false;
			if (fwrite(chunk.data(), 1, n, fout) != n)
				return false;

			nbytes -= n;
		}
		return true;
	}

	// Name of the scratch file holding one particle image. Defined once so the
	// writer and the reader of the cache cannot drift apart.
	FileName scratchParticleFile(const FileName &fn_scratch, int optics_group, long int og_idx,
	                             bool is_3D, bool is_tomo)
	{
		const char *ext = is_3D ? ".mrc" : (is_tomo ? ".mrcs" : ".mrc");
		return fn_scratch + "opticsgroup" + integerToString(optics_group + 1)
		     + "_particle" + integerToString((int) (og_idx + 1)) + ext;
	}

	// Glob prefix shared by all particle files of an optics group.
	FileName scratchParticlePrefix(const FileName &fn_scratch, int optics_group)
	{
		return fn_scratch + "opticsgroup" + integerToString(optics_group + 1) + "_particle";
	}

	// One raw byte copy of a particle image from its source into scratch.
	struct CopyJob
	{
		int optics_group;
		long int og_idx;      // scratch-file index within the optics group
		FileName fn_src;      // source .mrcs stack (SPA 2D) or .mrc/.mrcs file (3D/tomo)
		size_t byte_offset;   // byte offset in fn_src for this image
		size_t byte_size;     // bytes of payload to copy
		bool has_header;      // true: a patched 1024-byte header is prepended at write time
		bool swapped;         // true: source header is big-endian on this machine
	};

	// Builds exactly one job per distinct scratch file, in scratch-file index
	// order. The indices come from Experiment::assignScratchFileIndices(), which
	// the caller has already run on every rank. That assigns each new source the
	// next free index in particle order and gives duplicates the index of their
	// first occurrence, so walking the particles in order sees each index either
	// repeated (duplicate) or exactly one past the highest seen (new file).
	// Recognising duplicates that way keeps a single source of truth for the
	// mapping instead of re-deriving the "same source" rule here.
	std::vector<CopyJob> buildCopyJobs(const std::vector<ExpParticle> &particles,
	                                   bool is_3D, bool is_tomo, int nr_optics_groups)
	{
		std::vector<CopyJob> jobs;
		// Cache headers so each source stack is opened only once.
		std::map<FileName, MrcHeaderInfo> header_cache;
		// Highest index emitted so far, per optics group.
		std::vector<long int> next_idx(nr_optics_groups, 0);

		for (long int part_id = 0; part_id < (long int) particles.size(); part_id++)
		{
			const int optics_group = particles[part_id].optics_group;
			const long int og_idx = particles[part_id].optics_group_id;

			// A duplicate carries the index of its first occurrence, which has
			// already produced a job pointing at the same file.
			if (og_idx < next_idx[optics_group])
				continue;
			// Anything else means the numbering no longer matches particle
			// order; silently continuing would write the wrong file.
			if (og_idx != next_idx[optics_group])
				REPORT_ERROR("BUG: scratch indices are not in particle order for optics group "
				             + integerToString(optics_group + 1));
			next_idx[optics_group] = og_idx + 1;

			FileName fn_img = particles[part_id].name;
			long int imgno;
			FileName fn_stack;
			fn_img.decompose(imgno, fn_stack);
			// fn_stack is the file on disk; for names without "@index" it is
			// simply fn_img itself.
			const FileName &fn_src = fn_stack;

			// Determine the real on-disk representation from the MRC header.
			// Never assume float32: particle stacks are commonly float16.
			MrcHeaderInfo hdr;
			auto it_hdr = header_cache.find(fn_src);
			if (it_hdr == header_cache.end())
			{
				if (!readMrcHeader(fn_src, hdr))
					REPORT_ERROR("ERROR: cannot read a valid MRC header from " + fn_src);
				header_cache[fn_src] = hdr;
			}
			else
			{
				hdr = it_hdr->second;
			}

			if (is_3D && hdr.nz <= 1)
				REPORT_ERROR("BUG: inconsistent is_3D values!");

			// SPA particle names always carry "@index"; a bare name is a
			// single-image file and means its first (only) image.
			if (!is_3D && !is_tomo && imgno < 1)
				imgno = 1;

			// Subtomogram particles are consumed as a whole stack (the reader
			// takes img() in full), so the file is copied verbatim. SPA 2D
			// particles are one image inside a shared .mrcs stack.
			const bool whole_file = (is_3D || is_tomo);

			CopyJob job;
			job.optics_group = optics_group;
			job.og_idx = og_idx;
			job.byte_offset = 0;
			job.byte_size = 0;
			job.has_header = false;
			job.swapped = hdr.swapped;

			// Validate the source mode once and get its per-image byte count.
			size_t image_bytes;
			if (!mrcBytesPerImage(hdr, image_bytes))
				REPORT_ERROR("ERROR: unsupported MRC mode in " + fn_src);

			if (whole_file)
			{
				// Copy the whole file verbatim (header included). The on-disk
				// length is authoritative; the header is only used for the
				// dimensionality sanity check above.
				job.fn_src = fn_src;
				job.byte_size = fileSize(fn_src);
				if (job.byte_size < 1024)
					REPORT_ERROR("ERROR: cannot determine a valid size for " + fn_src);
			}
			else
			{
				// Extract a single image from the stack as raw bytes
				if (imgno > hdr.nz)
					REPORT_ERROR("ERROR: image number " + integerToString(imgno)
					             + " exceeds stack size " + integerToString(hdr.nz)
					             + " of " + fn_src);

				job.fn_src = fn_stack;
				job.byte_size = image_bytes;
				job.has_header = true; // header is prepended at write time
				job.byte_offset = 1024 + (size_t) hdr.nsymbt
				                + (size_t)(imgno - 1) * image_bytes;

				const size_t stack_size = fileSize(fn_stack);
				if (stack_size > 0 && job.byte_offset + image_bytes > stack_size)
					REPORT_ERROR("ERROR: stack " + fn_stack + " is smaller than its header claims ("
					             + integerToString((long int) stack_size) + " bytes)");
			}

			jobs.push_back(job);
		}

		return jobs;
	}

	// Number of leading jobs whose bytes fit into the scratch space budget.
	// Computed identically on every rank so the published indices agree.
	long int jobsThatFit(const std::vector<CopyJob> &jobs, long int max_space)
	{
		long int total_bytes = 0;
		long int n = 0;

		for (long int i = 0; i < (long int) jobs.size(); i++)
		{
			const long int job_bytes = (long int) jobs[i].byte_size
			                         + (jobs[i].has_header ? 1024 : 0);
			if (total_bytes + job_bytes > max_space)
				break;

			total_bytes += job_bytes;
			n = i + 1;
		}

		return n;
	}

	// Copies one job to fn_out. Returns false on any I/O failure; the caller
	// removes the partial output file.
	bool copyOneJob(const CopyJob &job, const FileName &fn_out, std::vector<char> &buf)
	{
		const size_t bsize = job.byte_size;
		// 2D needs its patched header in front of the payload; 3D/tomo files
		// already carry theirs and are streamed through the chunk buffer.
		const size_t hdr_size = job.has_header ? 1024 : 0;
		const size_t total_size = hdr_size + bsize;
		if (hdr_size > 0 && buf.size() < total_size)
			buf.resize(total_size);

		FILE *fin = fopen(job.fn_src.c_str(), "rb");
		if (!fin)
		{
			std::cerr << " Warning: cannot open " << job.fn_src << " for scratch copy, skipping." << std::endl;
			return false;
		}

		bool ok = true;
		FILE *fout = NULL;

		if (hdr_size > 0)
		{
			// Patch the source stack header into a single-image header. The mode
			// field (offset 12) is deliberately left untouched so float16 data
			// stays float16 on scratch. Fields are written in the source
			// header's byte order.
			if (fread(buf.data(), 1, 1024, fin) != 1024)
			{
				std::cerr << " Warning: cannot read MRC header from " << job.fn_src << std::endl;
				ok = false;
			}
			else
			{
				storeInt32(buf.data() +  8, 1, job.swapped); // nz = 1
				storeInt32(buf.data() + 36, 1, job.swapped); // mz = 1
				storeInt32(buf.data() + 88, 0, job.swapped); // ispg = 0 (single 2D image)
				storeInt32(buf.data() + 92, 0, job.swapped); // nsymbt = 0 (drop ext header)
			}
		}

		if (ok && job.byte_offset > 0
		    && fseek(fin, (long int) job.byte_offset, SEEK_SET) != 0)
		{
			std::cerr << " Warning: cannot seek in " << job.fn_src << ", skipping." << std::endl;
			ok = false;
		}

		if (ok)
		{
			fout = fopen(fn_out.c_str(), "wb");
			if (!fout)
			{
				std::cerr << " Warning: cannot write to " << fn_out << ", skipping." << std::endl;
				ok = false;
			}
		}

		if (ok && hdr_size > 0)
		{
			if (fread(buf.data() + 1024, 1, bsize, fin) != bsize)
			{
				std::cerr << " Warning: short read from " << job.fn_src << ", skipping." << std::endl;
				ok = false;
			}
			else if (fwrite(buf.data(), 1, total_size, fout) != total_size)
			{
				std::cerr << " Warning: short write to " << fn_out << ", skipping." << std::endl;
				ok = false;
			}
		}
		else if (ok)
		{
			// 3D/tomo: stream the file so a big subtomogram is never held in
			// memory in full.
			if (!copyBytes(fin, fout, bsize, buf))
			{
				std::cerr << " Warning: copy of " << job.fn_src
				          << " to " << fn_out << " failed, skipping." << std::endl;
				ok = false;
			}
		}

		fclose(fin);
		if (fout)
			fclose(fout);

		return ok;
	}

	// Copies the first n_copy jobs in parallel. Returns true if the run was
	// aborted through the pipeline-control file. job_failed[i] is set for every
	// job whose output file could not be written.
	bool copyJobsToScratch(const std::vector<CopyJob> &jobs, long int n_copy,
	                       const FileName &fn_scratch, bool is_3D, bool is_tomo,
	                       std::vector<char> &job_failed)
	{
		const int check_abort_frequency = 100;
		std::atomic<bool> abort_copy(false);
		// pipeline_control_check_abort_job() must not be called concurrently.
		// With a dynamic schedule no single thread sees every iteration, but
		// fetch_add hands out unique values, so exactly one thread sees each
		// multiple of the frequency and polls on behalf of all of them.
		std::atomic<long int> tasks_started(0);

		#pragma omp parallel
		{
			// Thread-local buffer reused across jobs to keep memory pressure low.
			std::vector<char> buf;

			#pragma omp for schedule(dynamic, 64)
			for (long int i = 0; i < n_copy; i++)
			{
				if (abort_copy.load())
					continue;

				const long int started = tasks_started.fetch_add(1);
				if (started > 0 && started % check_abort_frequency == 0
				    && pipeline_control_check_abort_job())
				{
					abort_copy.store(true);
					continue;
				}

				const CopyJob &job = jobs[i];
				const FileName fn_out = scratchParticleFile(
						fn_scratch, job.optics_group, job.og_idx, is_3D, is_tomo);

				if (!copyOneJob(job, fn_out, buf))
				{
					// Never leave a truncated file behind: a later --reuse_scratch
					// run would otherwise treat it as valid cached data.
					remove(fn_out.c_str());
					job_failed[i] = 1;
				}
			}
		}

		return abort_copy.load();
	}

	// Scratch counts for the rank that performed the copy. getImageNameOnScratch()
	// treats "optics_group_id < nr_parts_on_scratch[og]" as "is cached", so this
	// must be the contiguous prefix of files that actually made it onto disk:
	// a gap would make later indices address a missing (or wrong) file.
	std::vector<long int> contiguousScratchCounts(const std::vector<CopyJob> &jobs, long int n_copy,
	                                              const std::vector<char> &job_failed,
	                                              int nr_optics_groups)
	{
		std::vector<long int> counts(nr_optics_groups, 0);
		std::vector<char> closed(nr_optics_groups, 0);

		for (long int i = 0; i < n_copy; i++)
		{
			const int og = jobs[i].optics_group;
			if (closed[og])
				continue;

			if (job_failed[i] || jobs[i].og_idx != counts[og])
				closed[og] = 1;
			else
				counts[og]++;
		}

		return counts;
	}
}

bool Experiment::getImageNameOnScratch(long int part_id, FileName &fn_img, bool is_ctf_image)
{
	// CTF volumes are not copied to scratch; callers fall back to the original.
	if (is_ctf_image)
		return false;

	int optics_group = getOpticsGroup(part_id);
    //TODO: move optics_group_id to particle, not image!!!!
	long int my_id = particles[part_id].optics_group_id;

    //REPORT_ERROR("DEBUG: STILL NEED TO ACCOUNT FOR MULTIPLE IMAGES PER PARTICLE HEREE!!!!!! UNFINISHED CODE....");

#ifdef DEBUG_SCRATCH
	std::cerr << "part_id = " << part_id << " img_id = " << img_id << " my_id = " << my_id << " nr_parts_on_scratch[" << optics_group << "] = " << nr_parts_on_scratch[optics_group] << std::endl;
#endif

	if (fn_scratch != "" && my_id < nr_parts_on_scratch[optics_group])
	{
		fn_img = scratchParticleFile(fn_scratch, optics_group, my_id, is_3D, is_tomo);

#ifdef DEBUG_SCRATCH
		std::cerr << "getImageNameOnScratch: " << particles[part_id].name << " is cached at " << fn_img << std::endl;
#endif
		return true;
	}
	else
	{
		return false;
	}
}

void Experiment::assignScratchFileIndices()
{
	// Same numbering that copyParticlesToScratch() uses when it writes files.
	std::vector<long int> og_idx_counter(numberOfOpticsGroups(), 0);
	std::map<std::pair<FileName,int>, long int> seen_images;

	for (long int part_id = 0; part_id < (long int) particles.size(); part_id++)
	{
		auto key = std::make_pair(particles[part_id].name, particles[part_id].optics_group);
		auto it = seen_images.find(key);
		if (it != seen_images.end())
		{
			particles[part_id].optics_group_id = it->second;
		}
		else
		{
			long int idx = og_idx_counter[particles[part_id].optics_group]++;
			particles[part_id].optics_group_id = idx;
			seen_images[key] = idx;
		}
	}
}

void Experiment::setScratchDirectory(FileName _fn_scratch, bool do_reuse_scratch, int verb)
{
	// Make sure fn_scratch ends with a slash
	if (_fn_scratch[_fn_scratch.length()-1] != '/')
		_fn_scratch += '/';
	fn_scratch = _fn_scratch + "relion_volatile/";

	// All ranks, including those that never copy, need the same
	// particle-to-file mapping as the rank that wrote the files.
	assignScratchFileIndices();

	if (do_reuse_scratch)
	{
		nr_parts_on_scratch.resize(numberOfOpticsGroups(), 0);
		for (int optics_group = 0; optics_group < numberOfOpticsGroups(); optics_group++)
		{
			// The same name parts the writer uses, so "[0-9]*" cannot pick up
			// anything but this group's per-particle files.
			const FileName prefix = scratchParticlePrefix(fn_scratch, optics_group);
			const char *ext = is_3D ? ".mrc" : (is_tomo ? ".mrcs" : ".mrc");

			FileName fn_tmp = prefix + "[0-9]*" + ext;
			std::vector<FileName> fn_all;
			fn_tmp.globFiles(fn_all, true);
			std::set<FileName> present(fn_all.begin(), fn_all.end());

			// Count only the contiguous run starting at index 1: an interrupted
			// earlier copy can leave gaps, and getImageNameOnScratch() addresses
			// files by index, not by how many exist.
			long int nr_expected = 0;
			for (long int part_id = 0; part_id < (long int) particles.size(); part_id++)
				if (particles[part_id].optics_group == optics_group
				    && particles[part_id].optics_group_id >= nr_expected)
					nr_expected = particles[part_id].optics_group_id + 1;

			long int n = 0;
			while (n < nr_expected && present.count(scratchParticleFile(fn_scratch, optics_group, n, is_3D, is_tomo)))
				n++;

			nr_parts_on_scratch[optics_group] = n;
#ifdef DEBUG_SCRATCH
			if (verb > 0)
				std::cerr << " optics_group= " << (optics_group + 1) << " nr_parts_on_scratch[optics_group]= " << nr_parts_on_scratch[optics_group] << std::endl;
#endif
		}
	}
}

FileName Experiment::initialiseScratchLock(FileName _fn_scratch, FileName _fn_out)
{
	// Get a unique lockname for this run
	int uniqnr = rand() % 100000;
	FileName fn_uniq = _fn_out;
	fn_uniq.replaceAllSubstrings("/", "_");
	fn_uniq += "_lock" + integerToString(uniqnr);
	FileName fn_lock = fn_scratch + fn_uniq;

	if (exists(fn_lock))
		remove(fn_lock.c_str());

	return fn_lock;
}

bool Experiment::prepareScratchDirectory(FileName _fn_scratch, FileName fn_lock)
{
	if (fn_lock != "" && exists(fn_lock))
	{
		// Still measure how much free space there is
		struct statvfs vfs;
		statvfs(_fn_scratch.c_str(), &vfs);
		char nodename[64] = "undefined";
		gethostname(nodename,sizeof(nodename));
		std::string myhost(nodename);
		free_space_Gb = (RFLOAT)vfs.f_bsize * vfs.f_bfree / (1024 * 1024 * 1024);

		return false;
	}
	else
	{
		// Wipe the directory clean and make a new one
		std::string command;
		deleteDataOnScratch();

		// Make the scratch directory with write permissions
		command = "install -d -m 0777 " + fn_scratch;
		if (system(command.c_str()))
			REPORT_ERROR("ERROR: cannot execute: " + command);

		// Touch the lock file
		if(fn_lock != "")
		{
			touch(fn_lock);
			command = "chmod 0777 " + fn_lock;
			if (system(command.c_str()))
				REPORT_ERROR("ERROR: cannot execute: " + command);
		}

		// Measure how much free space there is
		struct statvfs vfs;
		statvfs(_fn_scratch.c_str(), &vfs);
		char nodename[64] = "undefined";
		gethostname(nodename,sizeof(nodename));
		std::string myhost(nodename);
		free_space_Gb = (RFLOAT)vfs.f_bsize * vfs.f_bfree / (1024 * 1024 * 1024);
		std::cout << " + On host " << myhost << ": free scratch space = " << free_space_Gb << " Gb." << std::endl;

		return true;
	}
}

void Experiment::deleteDataOnScratch()
{
	// Wipe the scratch directory
	if (fn_scratch != "" && exists(fn_scratch))
	{
		std::string command = " rm -rf " + fn_scratch;
		if (system(command.c_str()))
			REPORT_ERROR("ERROR: cannot execute: " + command);
	}
}

void Experiment::copyParticlesToScratch(int verb, bool do_copy, bool also_do_ctf_image, RFLOAT keep_free_scratch_Gb)
{

    // This function relies on prepareScratchDirectory() being called before!

	// CTF volumes are intentionally left where they are: only particle images
	// are cached. The parameter is kept for call-site compatibility.
	(void) also_do_ctf_image;

	const long int max_space = (free_space_Gb - keep_free_scratch_Gb) * 1024 * 1024 * 1024; // in bytes
	const int nr_og = numberOfOpticsGroups();
#ifdef DEBUG_SCRATCH
	std::cerr << " free_space_Gb = " << free_space_Gb << " GB, keep_free_scratch_Gb = " << keep_free_scratch_Gb << " GB.\n";
	std::cerr << " Max space RELION can use = " << max_space << " bytes" << std::endl;
#endif

	nr_parts_on_scratch.resize(nr_og, 0);
	// Every rank must agree on the particle-to-file mapping, so the indices
	// are derived here from the particle list alone.
	assignScratchFileIndices();

	// Phase 0: one job per distinct scratch file (CTF volumes are not copied).
	const std::vector<CopyJob> jobs = buildCopyJobs(particles, is_3D, is_tomo, nr_og);
	const long int total_jobs = (long int) jobs.size();

	// Phase 1: how many of them fit into the space we are allowed to use.
	const long int n_copy = jobsThatFit(jobs, max_space);
	if (n_copy < total_jobs && do_copy)
	{
		char nodename[64] = "undefined";
		gethostname(nodename, sizeof(nodename));
		std::string myhost(nodename);
		std::cerr << " Warning: scratch space full on " << myhost
		          << ". Remaining " << (total_jobs - n_copy)
		          << " particles will be read from where they were." << std::endl;
	}

	if (verb > 0 && do_copy)
	{
		std::cout << " Copying " << n_copy << " particles to scratch directory: " << fn_scratch << std::endl;
		init_progress_bar(total_jobs);
	}

	// Phase 2: the copy itself, done by the rank that owns the scratch disk.
	std::vector<char> job_failed;
	bool aborted = false;
	if (do_copy)
	{
		job_failed.assign(n_copy, 0);
		aborted = copyJobsToScratch(jobs, n_copy, fn_scratch, is_3D, is_tomo, job_failed);
	}

	if (aborted)
		exit(RELION_EXIT_ABORTED);

	// Phase 3: publish the per-optics-group counts the reader relies on.
	if (do_copy)
	{
		nr_parts_on_scratch = contiguousScratchCounts(jobs, n_copy, job_failed, nr_og);
	}
	else
	{
		// Not the copying rank: publish the full count, as before. The read
		// path verifies that the file exists before using it.
		for (long int i = 0; i < total_jobs; i++)
			nr_parts_on_scratch[jobs[i].optics_group]++;
	}

	long int total_copied = 0;
	for (int og = 0; og < nr_og; og++)
		total_copied += nr_parts_on_scratch[og];

	if (verb)
	{
		progress_bar(total_jobs);
		for (int i = 0; i < nr_og; i++)
		{
			std::cout << " For optics_group " << (i + 1) << ", there are " << nr_parts_on_scratch[i] << " particles on the scratch disk." << std::endl;
		}
	}

	if (do_copy && total_copied > 1)
	{
		std::string command = " chmod -R 777 " + fn_scratch + "/";
		if (system(command.c_str()))
			REPORT_ERROR("ERROR in executing: " + command);
	}
}
