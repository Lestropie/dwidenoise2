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
// The single argument is resolved as either:
// - the name of a schedule bundled with the software
//   (resolved to "<bundled directory>/<command>/<name>.txt"); or
// - the filesystem path to a user-authored schedule file.
extern const App::Option schedule_file_option;

// Multi-paragraph DESCRIPTION text documenting the schedule file format;
//   attached to each command's DESCRIPTION in the manner of the various
//   Denoise::*_description / Kernel::*_description constants.
extern const char *schedule_file_description;

// Whether the user has requested a manual schedule via -schedule_file.
bool requested();

// Resolve, read, parse and validate the schedule requested via -schedule_file.
// "command" is the name of the invoking command (e.g. "dwidenoise2"),
//   used both to locate command-specific bundled schedules and in messages.
// Precondition: requested() is true.
std::vector<Iterative::Iteration> load(const std::string &command);

} // namespace MR::Denoise::Schedule
