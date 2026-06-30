/* Required Notice: Copyright (c) 2025 Robert E. Smith <robert.smith@florey.edu.au>;
 * Required Notice: The Florey Institute of Neuroscience and Mental Health.
 *
 * Licensed under the PolyForm Noncommercial License 1.0.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     https://polyformproject.org/licenses/noncommercial/1.0.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 * either express or implied.
 * See the License of the specific language
 * governing permissions and limitations under the License.
 */

#include "denoise/schedule.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "denoise/denoise.h"
#include "exception.h"
#include "mrtrix.h"
#include "platform.h"

namespace MR::Denoise::Schedule {

using namespace App;

namespace {

// Recognised per-iteration column names.
// To add a new per-iteration variable in future:
//   1. add a field to Iterative::Iteration (denoise/iterative.h);
//   2. add its name here and a parsing branch in parse_row() below,
//      together with the default assigned in parse_row();
//   3. (optionally) add the column to the bundled schedule files.
// Existing schedule files that omit the new column will continue to parse,
//   taking the default; hence the format is forward-compatible.
const std::vector<std::string> recognised_columns(
    {"spatial_subsample", "kernel", "smooth_noise", "temporal_subsample", "update_noise",
     "partitions", "max_partition_size"});

// Whitespace-tokenise a line (collapsing runs of whitespace, dropping empties).
std::vector<std::string> tokenise(const std::string &line) {
  std::vector<std::string> result;
  std::istringstream stream(line);
  std::string token;
  while (stream >> token)
    result.push_back(token);
  return result;
}

// Strip a '#'-introduced comment from a line.
std::string strip_comment(const std::string &line) {
  const auto hash = line.find('#');
  return hash == std::string::npos ? line : line.substr(0, hash);
}

// "smooth_noise" is a boolean: "true" smooths the iteration's noise map estimate before it is
//   passed to the next iteration, "false" leaves it unsmoothed. Parsed via MR::to<bool>(), which
//   also accepts "yes"/"no" and 0/1; a column-specific message is raised on anything else.
noise_smooth_type parse_smooth_noise(const std::string &value, const size_t lineno) {
  bool smooth = false;
  try {
    smooth = to<bool>(value);
  } catch (Exception &) {
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value for column \"smooth_noise\" must be \"true\" or \"false\" (got \"" + value + "\")");
  }
  return smooth ? noise_smooth_type::SMOOTH : noise_smooth_type::NONE;
}

// The "kernel" column selects the per-iteration kernel type and its free parameter. Recognised
//   values (the size multiplier of the former "kernel_multiplier" column is achieved instead via
//   these per-kernel parameters):
//   - "aspect=<ratio>" (or "aspect_ratio=<ratio>"): spherical kernel of ~ratio*m voxels (rank-naive).
//   - "rmse=<tolerance>": spherical kernel grown until the estimator's predicted relative noise
//                         RMSE meets <tolerance> (a small positive fraction), floored at n>=m+r.
//   - "rank": rank-adaptive spherical kernel grown until n>=m+r (square noise block); no parameter.
Kernel::KernelSpec parse_kernel(const std::string &value, const size_t lineno) {
  const auto eq = value.find('=');
  const std::string key = (eq == std::string::npos) ? value : value.substr(0, eq);
  const std::string arg = (eq == std::string::npos) ? std::string() : value.substr(eq + 1);
  const auto parse_pos_double = [&](const std::string &what) {
    default_type r = 0.0;
    size_t consumed = 0;
    try {
      r = std::stod(arg, &consumed);
    } catch (std::exception &) {
      consumed = 0;
    }
    if (arg.empty() || consumed != arg.size() || !(r > 0.0))
      throw Exception("Schedule file line " + str(lineno) + ": "
                      "kernel " + what + " must be a positive number (got \"" + value + "\")");
    return r;
  };
  Kernel::KernelSpec spec;
  if (key == "aspect" || key == "aspect_ratio") {
    spec.type = Kernel::kernel_spec_type::ASPECT_RATIO;
    spec.param = parse_pos_double("aspect ratio (e.g. \"aspect=2.0\")");
  } else if (key == "rmse") {
    spec.type = Kernel::kernel_spec_type::RMSE;
    spec.param = parse_pos_double("RMSE tolerance (e.g. \"rmse=0.02\")");
    if (spec.param >= 1.0)
      throw Exception("Schedule file line " + str(lineno) + ": "
                      "kernel RMSE tolerance must be a small fraction below 1 (got \"" + value + "\")");
  } else if (key == "rank") {
    if (!arg.empty())
      throw Exception("Schedule file line " + str(lineno) + ": "
                      "kernel \"rank\" takes no parameter (got \"" + value + "\")");
    spec.type = Kernel::kernel_spec_type::RANK;
  } else {
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "unrecognised kernel \"" + value + "\"; valid kernels are "
                    "\"aspect=<ratio>\", \"rmse=<tolerance>\" and \"rank\"");
  }
  return spec;
}

std::array<ssize_t, 3> parse_spatial_subsample(const std::string &value, const size_t lineno) {
  std::vector<ssize_t> factors;
  try {
    factors = parse_ints<ssize_t>(value);
  } catch (Exception &) {
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value \"" + value + "\" for column \"spatial_subsample\" must be an integer "
                    "or a comma-separated triplet of integers");
  }
  std::array<ssize_t, 3> result{};
  if (factors.size() == 1)
    result = {factors[0], factors[0], factors[0]};
  else if (factors.size() == 3)
    result = {factors[0], factors[1], factors[2]};
  else
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value for column \"spatial_subsample\" must be a single integer "
                    "or a comma-separated triplet of integers (got " + str(factors.size()) + " values)");
  for (ssize_t axis = 0; axis != 3; ++axis) {
    if (result[axis] < 1)
      throw Exception("Schedule file line " + str(lineno) + ": "
                      "spatial subsampling factors must be positive integers");
  }
  return result;
}

default_type parse_temporal_subsample(const std::string &value, const size_t lineno) {
  default_type result = 0.0;
  size_t consumed = 0;
  try {
    result = std::stod(value, &consumed);
  } catch (std::exception &) {
    consumed = 0;
  }
  if (consumed != value.size())
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value \"" + value + "\" for column \"temporal_subsample\" is not a valid number");
  if (!(result > 0.0 && result <= 1.0))
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value for column \"temporal_subsample\" must lie within (0.0, 1.0] (got \"" + value + "\")");
  return result;
}

bool parse_update_noise(const std::string &value, const size_t lineno) {
  if (value == "true" || value == "1")
    return true;
  if (value == "false" || value == "0")
    return false;
  throw Exception("Schedule file line " + str(lineno) + ": "
                  "value for column \"update_noise\" must be one of \"true\" or \"false\" (got \"" + value + "\")");
}

// Parse a non-negative integer column value, throwing a column-specific message on failure.
ssize_t parse_nonneg_int(const std::string &value, const std::string &column, const size_t lineno) {
  ssize_t result = 0;
  size_t consumed = 0;
  try {
    result = ssize_t(std::stoll(value, &consumed));
  } catch (std::exception &) {
    consumed = 0;
  }
  if (consumed != value.size())
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value \"" + value + "\" for column \"" + column + "\" is not a valid integer");
  return result;
}

// Explicit partition count P; must be >= 1 (1 ⇒ no partitioning).
ssize_t parse_partitions(const std::string &value, const size_t lineno) {
  const ssize_t result = parse_nonneg_int(value, "partitions", lineno);
  if (result < 1)
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value for column \"partitions\" must be a positive integer (got \"" + value + "\")");
  return result;
}

// Maximum volumes per partition; "none" / "0" ⇒ no limit (unset). Otherwise must be >= 1.
std::optional<ssize_t> parse_max_partition_size(const std::string &value, const size_t lineno) {
  if (value == "none")
    return std::nullopt;
  const ssize_t result = parse_nonneg_int(value, "max_partition_size", lineno);
  if (result == 0)
    return std::nullopt;
  if (result < 1)
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value for column \"max_partition_size\" must be a positive integer or \"none\" "
                    "(got \"" + value + "\")");
  return result;
}

// Directory in which command-specific bundled schedules reside, holding both the command's
//   "default" schedule and any other named schedules (e.g. "fast"). The schedule files are
//   installed by the MRtrix3 cmake build alongside the other MRtrix3 shared data files, under
//   "<datadir>/mrtrix3/<command>/". They are located here relative to the running executable:
//   in the build tree the executables live in "<build>/bin/" and the shared data in
//   "<build>/share/mrtrix3/", so the schedules reside at "<exe dir>/../share/mrtrix3/<command>/".
// Platform::get_executable_path() throws if the executable location cannot be determined; that
//   propagates as a clear error, as the command's default schedule is now loaded from this
//   location rather than being hard-coded.
std::string bundled_directory(const std::string &command) {
  const std::filesystem::path executable_path = Platform::get_executable_path();
  return (executable_path.parent_path().parent_path() / "share" / "mrtrix3" / command).string();
}

// Human-readable list of the bundled schedules available for "command",
//   for inclusion in an error message when resolution fails.
std::string available_bundled(const std::string &command) {
  std::vector<std::string> names;
  try {
    const std::filesystem::path dir(bundled_directory(command));
    if (std::filesystem::is_directory(dir)) {
      for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
          names.push_back(entry.path().stem().string());
      }
    }
  } catch (...) {
    // Best-effort only; fall through to whatever was collected.
  }
  if (names.empty())
    return " (no bundled schedules were found for this command)";
  std::sort(names.begin(), names.end());
  return " (bundled schedules available for " + command + ": " + join(names, ", ") + ")";
}

// Resolve a -schedule argument to an on-disk path. The argument is interpreted, in order:
//   1. as a filesystem path (absolute or relative to the working directory); if it names an
//      existing file it is used verbatim;
//   2. otherwise as the name of a schedule bundled with the software, residing in the same
//      directory that supplies the command's default schedule (bundled_directory). The ".txt"
//      extension is appended only when the supplied name carries no extension of its own, so
//      that both "legacy" and "legacy.txt" resolve to the bundled "legacy.txt".
// An Exception is thrown only when neither interpretation locates an existing file.
std::string resolve(const std::string &spec, const std::string &command) {
  if (std::ifstream(spec).good())
    return spec;
  std::filesystem::path name(spec);
  if (!name.has_extension())
    name += ".txt";
  const std::string candidate = (std::filesystem::path(bundled_directory(command)) / name).string();
  if (std::ifstream(candidate).good())
    return candidate;
  throw Exception("Noise estimation schedule \"" + spec + "\" "
                  "is neither the path to an existing file "
                  "nor the name of a bundled schedule" + available_bundled(command));
}

// Total volume count above which use of a command's embedded default schedule prompts a
//   "this may be slow" warning. The default schedule's coarsest passes use large
//   patches whose PCA cost grows with the volume count;
//   beyond a few hundred volumes a lighter schedule is usually preferable.
constexpr size_t default_schedule_slow_threshold = 255;

std::vector<Iterative::Iteration> parse(const std::string &path, const std::string &command) {
  // dwi2noise (re)estimates the noise level in every iteration, so the "update_noise" column
  //   carries no meaning for it and is rejected outright rather than silently ignored.
  const bool permit_update_noise = (command != "dwi2noise");

  std::ifstream in(path);
  if (!in)
    throw Exception("Unable to open noise estimation schedule file \"" + path + "\"");

  std::vector<std::string> header;
  std::vector<Iterative::Iteration> result;
  std::string line;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    const std::vector<std::string> tokens = tokenise(strip_comment(line));
    if (tokens.empty())
      continue;

    // The first non-comment, non-blank line names the columns.
    if (header.empty()) {
      for (const auto &column : tokens) {
        if (std::find(recognised_columns.begin(), recognised_columns.end(), column) == recognised_columns.end())
          throw Exception("Schedule file \"" + path + "\" line " + str(lineno) + ": "
                          "unrecognised column \"" + column + "\"; "
                          "recognised columns are: " + join(recognised_columns, ", "));
        if (column == "update_noise" && !permit_update_noise)
          throw Exception("Schedule file \"" + path + "\" line " + str(lineno) + ": "
                          "column \"update_noise\" is not permitted in a " + command + " schedule; "
                          "the " + command + " command (re)estimates the noise level in every iteration");
        if (std::count(tokens.begin(), tokens.end(), column) > 1)
          throw Exception("Schedule file \"" + path + "\" line " + str(lineno) + ": "
                          "column \"" + column + "\" specified more than once");
      }
      header = tokens;
      continue;
    }

    if (tokens.size() != header.size())
      throw Exception("Schedule file \"" + path + "\" line " + str(lineno) + ": "
                      "expected " + str(header.size()) + " fields to match the header, but found " +
                      str(tokens.size()));

    // Defaults for any column not present in the header.
    //   temporal_subsample (1.0) and update_noise (unset) take their in-struct defaults;
    //   update_noise is resolved to a concrete value per command after loading.
    Iterative::Iteration iteration;
    iteration.spatial_subsample_ratios = {default_spatial_subsample_ratio, default_spatial_subsample_ratio, default_spatial_subsample_ratio};
    // Default kernel (when the "kernel" column is omitted): rank-naive aspect ratio n ~ 2m.
    iteration.kernel.type = Kernel::kernel_spec_type::ASPECT_RATIO;
    iteration.kernel.param = 2.0;
    iteration.smooth_noiseout = noise_smooth_type::NONE;
    for (size_t column = 0; column != header.size(); ++column) {
      const std::string &key = header[column];
      const std::string &value = tokens[column];
      if (key == "spatial_subsample")
        iteration.spatial_subsample_ratios = parse_spatial_subsample(value, lineno);
      else if (key == "kernel")
        iteration.kernel = parse_kernel(value, lineno);
      else if (key == "smooth_noise")
        iteration.smooth_noiseout = parse_smooth_noise(value, lineno);
      else if (key == "temporal_subsample")
        iteration.temporal_subsample = parse_temporal_subsample(value, lineno);
      else if (key == "update_noise")
        iteration.update_noise = parse_update_noise(value, lineno);
      else if (key == "partitions")
        iteration.num_partitions = parse_partitions(value, lineno);
      else if (key == "max_partition_size")
        iteration.max_partition_size = parse_max_partition_size(value, lineno);
    }
    // The two partition controls are mutually exclusive ways of specifying the partition
    //   count: a row may give an explicit count (partitions > 1) or a maximum partition
    //   size, but not both at once.
    if (iteration.num_partitions > 1 && iteration.max_partition_size.has_value())
      throw Exception("Schedule file \"" + path + "\" line " + str(lineno) + ": "
                      "columns \"partitions\" and \"max_partition_size\" are mutually exclusive; "
                      "specify at most one (set \"partitions\" to 1 or \"max_partition_size\" to "
                      "\"none\" to disable one of them on this row)");
    // Smoothing the iteration's noise map estimate is only meaningful when that iteration
    //   (re)estimates the noise level; a row that explicitly disables estimation cannot also
    //   request smoothing. (The complementary case in which update_noise is left to its
    //   per-command default is enforced by each command once that default is resolved.)
    if (iteration.smooth_noiseout == noise_smooth_type::SMOOTH &&
        iteration.update_noise.has_value() && !iteration.update_noise.value())
      throw Exception("Schedule file \"" + path + "\" line " + str(lineno) + ": "
                      "column \"smooth_noise\" may be true only when \"update_noise\" is also true "
                      "in the same iteration");
    result.push_back(iteration);
  }

  if (header.empty())
    throw Exception("Schedule file \"" + path + "\" contains no column header");
  if (result.empty())
    throw Exception("Schedule file \"" + path + "\" defines no iterations "
                    "(a column header but no data rows)");

  // The first iteration has no signal-rank density from a prior iteration, so it cannot use a
  //   rank-dependent kernel; it must use an aspect-ratio kernel (the default when "kernel" is
  //   omitted). "rank" and "rmse" kernels are valid only from the second iteration onward.
  if (result.front().kernel.type != Kernel::kernel_spec_type::ASPECT_RATIO)
    throw Exception("Schedule file \"" + path + "\": the first iteration must use an aspect-ratio "
                    "kernel (e.g. \"aspect=2.0\"); the \"rank\" and \"rmse\" kernels require a "
                    "signal-rank density from a prior iteration and so may not appear on the first row");

  // Any iteration that is not the last must (re)estimate the noise level: a non-final
  //   iteration exists precisely to refine the estimate fed to the next iteration, so a
  //   non-final "update_noise false" would make that iteration pointless.
  for (size_t i = 0; i + 1 < result.size(); ++i) {
    if (result[i].update_noise.has_value() && !result[i].update_noise.value())
      throw Exception("Schedule file \"" + path + "\": "
                      "column \"update_noise\" must be true for all but the final iteration "
                      "(iteration " + str(i + 1) + " has update_noise false)");
  }

  // dwidenoise2's final row is the reconstruction pass: its noise map is what is actually applied to
  //   the data (for the variance-stabilising transform and the denoising threshold), and is also what
  //   -noise_out exports. Smoothing it would make the exported / applied map deviate from the map the
  //   PCA actually used, so it is rejected here. (dwi2noise, by contrast, *does* smooth its final row
  //   when requested: that row produces the command's exported estimate, not a reconstruction.)
  if (command == "dwidenoise2" && result.back().smooth_noiseout == noise_smooth_type::SMOOTH)
    throw Exception("Schedule file \"" + path + "\": the final (reconstruction) row of a dwidenoise2 "
                    "schedule must not set smooth_noise true; smoothing the reconstruction noise map "
                    "would make the exported map deviate from the one actually applied to the data");

  return result;
}

} // namespace

const Option schedule_option =
    Option("schedule",
           "manually specify the multi-resolution iteration schedule"
           " used for noise level estimation,"
           " in place of the command's default schedule;"
           " the argument is either the name of a schedule bundled with the software"
           " or the path to a schedule file (see Description for the file format)")
    + Argument("name/file").type_text();

const char *schedule_description =
    "By default the noise level is estimated via an a priori multi-resolution iteration"
    " schedule. Option -schedule instead reads the schedule from a text file,"
    " which is useful for reproducibly applying a bespoke schedule across a cohort"
    " with minimal command-line entry. The argument is first interpreted as the path to such a"
    " file (absolute or relative to the working directory); if no file is found there it is"
    " instead interpreted as the name of one of the schedules bundled with the software,"
    " appending a \".txt\" extension if the name does not already carry one"
    " (the bundled \"default\" schedule reproduces the command's built-in default).";

bool requested() { return !get_options("schedule").empty(); }

std::vector<Iterative::Iteration> load(const std::string &command) {
  auto opt = get_options("schedule");
  assert(!opt.empty());
  const std::string spec(opt[0][0]);
  const std::string path = resolve(spec, command);
  std::vector<Iterative::Iteration> schedule = parse(path, command);
  INFO("Using noise estimation schedule from \"" + path + "\" (" + str(schedule.size()) + " iteration" +
       (schedule.size() == 1 ? "" : "s") + ")");
  return schedule;
}

std::vector<Iterative::Iteration> load_default(const std::string &command) {
  const std::string path = bundled_directory(command) + "/default.txt";
  if (!std::ifstream(path).good())
    throw Exception("Unable to locate the bundled default noise estimation schedule for command \"" + command +
                    "\" (expected at \"" + path + "\"); the software installation may be incomplete");
  std::vector<Iterative::Iteration> schedule = parse(path, command);
  INFO("Using bundled default noise estimation schedule \"" + path + "\" (" + str(schedule.size()) + " iteration" +
       (schedule.size() == 1 ? "" : "s") + ")");
  return schedule;
}

void warn_if_default_schedule_slow(const size_t num_volumes) {
  if (num_volumes > default_schedule_slow_threshold)
    WARN("Input data contain " + str(num_volumes) + " volumes; "
         "the command's default noise estimation schedule performs full multi-resolution estimation "
         "and may be slow for a dataset of this size. "
         "Consider specifying a lighter schedule via -schedule "
         "(e.g. the bundled \"vlarge\" schedule).");
}

} // namespace MR::Denoise::Schedule
