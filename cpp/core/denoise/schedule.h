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

#pragma once

#include <string>
#include <vector>

#include "app.h"
#include "denoise/iterative.h"

namespace MR::Denoise::Schedule {

// Command-line option through which a user may take manual control of the
//   multi-resolution iteration schedule used for noise level estimation,
//   in place of the command's a priori default schedule.
// The single argument is resolved (see schedule.cpp resolve()) as either:
// - the filesystem path to a user-authored schedule file (tried first); or, failing that,
// - the name of a schedule bundled with the software
//   (resolved to "<bundled directory>/<command>/<name>.txt", a ".txt" extension being
//   appended only when the supplied name carries none of its own).
extern const App::Option schedule_option;

// Multi-paragraph DESCRIPTION text documenting the schedule file format;
//   attached to each command's DESCRIPTION in the manner of the various
//   Denoise::*_description / Kernel::*_description constants.
extern const char *schedule_description;

// Whether the user has requested a manual schedule via -schedule.
bool requested();

// Resolve, read, parse and validate the schedule requested via -schedule.
// "command" is the name of the invoking command (e.g. "dwidenoise2"),
//   used to locate command-specific bundled schedules, to police command-specific column
//   rules (e.g. "update_noise" is rejected for dwi2noise), and in messages.
// Precondition: requested() is true.
std::vector<Iterative::Iteration> load(const std::string &command);

// Load the command's bundled "default" schedule, used when the user does not supply
//   -schedule. The schedule files are installed alongside the other MRtrix3 shared
//   data files and are located relative to the running executable (see schedule.cpp);
//   this replaces the schedule that was previously hard-coded into each command.
std::vector<Iterative::Iteration> load_default(const std::string &command);

// Issue a warning when a command's bundled default noise estimation schedule is used on a
//   dataset large enough that the full multi-resolution default may be slow, suggesting a
//   lighter -schedule. "num_volumes" is the total number of volumes implied by the input
//   (the product of all non-spatial dimensions). Called by the commands when no -schedule
//   was supplied and the bundled default schedule is therefore in effect.
void warn_if_default_schedule_slow(size_t num_volumes);

} // namespace MR::Denoise::Schedule
