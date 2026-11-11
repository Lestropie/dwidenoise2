/* Required Notice: Copyright (c) 2026 Robert E. Smith <robert.smith@florey.edu.au>;
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

#include "denoise/metadata.h"

#include <algorithm>
#include <cmath>

#include "exception.h"
#include "mrtrix.h"

namespace MR::Denoise {

const std::vector<std::string> descriptive_keys{"ImageType",
                                                "ImageTypeText",
                                                "ImageComments",
                                                "comments",
                                                "ImageHistory",
                                                "DerivationDescription",
                                                "SeriesDescription",
                                                "ProtocolName"};

std::vector<std::string> metadata_tokens(const std::string &value) {
  std::vector<std::string> result;
  for (const auto &token : split(value, " \t\n\r\\,;:[]()", true))
    result.push_back(uppercase(token));
  return result;
}

std::vector<std::string> find_metadata_tokens(const KeyValues &keyval,
                                              const std::vector<std::string> &keys,
                                              const std::vector<std::string> &tokens) {
  std::vector<std::string> result;
  for (const auto &key : keys) {
    const auto entry = keyval.find(key);
    if (entry == keyval.end())
      continue;
    const std::vector<std::string> value_tokens = metadata_tokens(entry->second);
    for (const auto &token : tokens) {
      if (std::find(value_tokens.begin(), value_tokens.end(), token) != value_tokens.end())
        result.push_back("\"" + key + "\" contains \"" + token + "\"");
    }
  }
  return result;
}

std::optional<std::string> metadata_string(const KeyValues &keyval, const std::string &key) {
  const auto entry = keyval.find(key);
  if (entry == keyval.end())
    return std::nullopt;
  const std::string value = strip(entry->second);
  if (value.empty())
    return std::nullopt;
  return value;
}

std::optional<bool> metadata_bool(const KeyValues &keyval, const std::string &key) {
  const auto entry = keyval.find(key);
  if (entry == keyval.end())
    return std::nullopt;
  try {
    return to<bool>(entry->second);
  } catch (Exception &) {
    return std::nullopt;
  }
}

std::optional<default_type> metadata_float(const KeyValues &keyval, const std::string &key) {
  const auto entry = keyval.find(key);
  if (entry == keyval.end())
    return std::nullopt;
  try {
    const default_type value = to<default_type>(entry->second);
    if (!std::isfinite(value))
      return std::nullopt;
    return value;
  } catch (Exception &) {
    return std::nullopt;
  }
}

std::optional<default_type> metadata_max(const KeyValues &keyval, const std::string &key) {
  const auto entry = keyval.find(key);
  if (entry == keyval.end())
    return std::nullopt;
  default_type result = 0.0;
  for (const auto &token : split(entry->second, " \t\n\r\\,;", true)) {
    try {
      const default_type value = to<default_type>(token);
      if (!std::isfinite(value))
        return std::nullopt;
      result = std::max(result, value);
    } catch (Exception &) {
      return std::nullopt;
    }
  }
  return result > 0.0 ? std::optional<default_type>(result) : std::nullopt;
}

} // namespace MR::Denoise
