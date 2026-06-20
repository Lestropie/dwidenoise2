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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <climits>
#include <unistd.h>

#include "denoise/denoise.h"
#include "exception.h"
#include "mrtrix.h"

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
const std::vector<std::string> recognised_columns({"subsample", "kernel_size", "smooth"});

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

noise_smooth_type parse_smooth(const std::string &value, const size_t lineno) {
  if (value == "smooth" || value == "1")
    return noise_smooth_type::SMOOTH;
  if (value == "none" || value == "0")
    return noise_smooth_type::NONE;
  throw Exception("Schedule file line " + str(lineno) + ": "
                  "value for column \"smooth\" must be one of \"none\" or \"smooth\" (got \"" + value + "\")");
}

default_type parse_kernel_size(const std::string &value, const size_t lineno) {
  default_type result = 0.0;
  size_t consumed = 0;
  try {
    result = std::stod(value, &consumed);
  } catch (std::exception &) {
    consumed = 0;
  }
  if (consumed != value.size())
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value \"" + value + "\" for column \"kernel_size\" is not a valid number");
  if (result <= 0.0)
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value for column \"kernel_size\" must be greater than zero (got \"" + value + "\")");
  return result;
}

std::array<ssize_t, 3> parse_subsample(const std::string &value, const size_t lineno) {
  std::vector<ssize_t> factors;
  try {
    factors = parse_ints<ssize_t>(value);
  } catch (Exception &) {
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value \"" + value + "\" for column \"subsample\" must be an integer "
                    "or a comma-separated triplet of integers");
  }
  std::array<ssize_t, 3> result{};
  if (factors.size() == 1)
    result = {factors[0], factors[0], factors[0]};
  else if (factors.size() == 3)
    result = {factors[0], factors[1], factors[2]};
  else
    throw Exception("Schedule file line " + str(lineno) + ": "
                    "value for column \"subsample\" must be a single integer "
                    "or a comma-separated triplet of integers (got " + str(factors.size()) + " values)");
  for (ssize_t axis = 0; axis != 3; ++axis) {
    if (result[axis] < 1)
      throw Exception("Schedule file line " + str(lineno) + ": "
                      "subsampling factors must be positive integers");
  }
  return result;
}

// Directory in which command-specific bundled schedules reside.
// Resolution order:
//   1. environment variable DWIDENOISE2_SCHEDULE_PATH (set by the container build);
//   2. a location relative to the executable (<exe dir>/../share/dwidenoise2);
//   3. a final relative-path fallback.
// The command name is appended as a subdirectory in all cases.
std::string bundled_directory(const std::string &command) {
  if (const char *const env = std::getenv("DWIDENOISE2_SCHEDULE_PATH"))
    return std::string(env) + "/" + command;
  char buffer[PATH_MAX];
  const ssize_t count = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (count > 0) {
    buffer[count] = '\0';
    const std::string exe(buffer);
    const auto slash = exe.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string(".") : exe.substr(0, slash);
    return dir + "/../share/dwidenoise2/" + command;
  }
  return "share/dwidenoise2/" + command;
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

// Resolve a -schedule_file argument to an on-disk path:
//   an existing file is used verbatim; otherwise it is treated as a bundled name.
std::string resolve(const std::string &spec, const std::string &command) {
  if (std::ifstream(spec).good())
    return spec;
  const std::string candidate = bundled_directory(command) + "/" + spec + ".txt";
  if (std::ifstream(candidate).good())
    return candidate;
  throw Exception("Noise estimation schedule \"" + spec + "\" "
                  "is neither the path to an existing file "
                  "nor the name of a bundled schedule" + available_bundled(command));
}

std::vector<Iterative::Iteration> parse(const std::string &path) {
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
    Iterative::Iteration iteration;
    iteration.subsample_ratios = {default_subsample_ratio, default_subsample_ratio, default_subsample_ratio};
    iteration.kernel_size_multiplier = 1.0;
    iteration.smooth_noiseout = noise_smooth_type::NONE;
    for (size_t column = 0; column != header.size(); ++column) {
      const std::string &key = header[column];
      const std::string &value = tokens[column];
      if (key == "subsample")
        iteration.subsample_ratios = parse_subsample(value, lineno);
      else if (key == "kernel_size")
        iteration.kernel_size_multiplier = parse_kernel_size(value, lineno);
      else if (key == "smooth")
        iteration.smooth_noiseout = parse_smooth(value, lineno);
    }
    result.push_back(iteration);
  }

  if (header.empty())
    throw Exception("Schedule file \"" + path + "\" contains no column header");
  if (result.empty())
    throw Exception("Schedule file \"" + path + "\" defines no iterations "
                    "(a column header but no data rows)");
  return result;
}

} // namespace

const Option schedule_file_option =
    Option("schedule_file",
           "manually specify the multi-resolution iteration schedule"
           " used for noise level estimation,"
           " in place of the command's default schedule;"
           " the argument is either the name of a schedule bundled with the software"
           " or the path to a schedule file (see Description for the file format)")
    + Argument("name/file").type_text();

const char *schedule_file_description =
    "By default the noise level is estimated via an a priori multi-resolution iteration"
    " schedule. Option -schedule_file instead reads the schedule from a text file,"
    " which is useful for reproducibly applying a bespoke schedule across a cohort"
    " with minimal command-line entry. The argument may be either the path to such a"
    " file, or the name of one of the schedules bundled with the software"
    " (the bundled \"default\" schedule reproduces the command's built-in default).";

bool requested() { return !get_options("schedule_file").empty(); }

std::vector<Iterative::Iteration> load(const std::string &command) {
  auto opt = get_options("schedule_file");
  assert(!opt.empty());
  const std::string spec(opt[0][0]);
  const std::string path = resolve(spec, command);
  std::vector<Iterative::Iteration> schedule = parse(path);
  INFO("Using noise estimation schedule from \"" + path + "\" (" + str(schedule.size()) + " iteration" +
       (schedule.size() == 1 ? "" : "s") + ")");
  return schedule;
}

} // namespace MR::Denoise::Schedule
