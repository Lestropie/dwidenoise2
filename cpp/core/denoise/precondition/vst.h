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

#pragma once

#include <type_traits>

#include "denoise/precondition/noise_model/base.h"
#include "types.h"

// Pointwise application of the variance-stabilising transform (and its inverses) to a single
//   datum, dispatched by SFINAE so the compiler does not attempt to build the complex-valued
//   overloads for real types (and vice versa). For complex (Gaussian) data the scalar transform
//   is applied independently to the real and imaginary channels, as documented by the
//   NoiseModel::Base interface.
namespace MR::Denoise::Precondition {

// Forward variance-stabilising transform applied to a single datum.
template <typename T>
typename std::enable_if<!is_complex<T>::value, T>::type
vst_forward(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return T(model.stabilise(default_type(in), sigma));
}
template <typename T>
typename std::enable_if<is_complex<T>::value, T>::type
vst_forward(const NoiseModel::Base &model, const T in, const default_type sigma) {
  using R = typename T::value_type;
  return T(R(model.stabilise(default_type(in.real()), sigma)), R(model.stabilise(default_type(in.imag()), sigma)));
}

// Algebraic inverse of the forward variance-stabilising transform,
//   recovering the conventional (still-biased) intensity scale.
template <typename T>
typename std::enable_if<!is_complex<T>::value, T>::type
vst_inverse(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return T(model.inverse_algebraic(default_type(in), sigma));
}
template <typename T>
typename std::enable_if<is_complex<T>::value, T>::type
vst_inverse(const NoiseModel::Base &model, const T in, const default_type sigma) {
  using R = typename T::value_type;
  return T(R(model.inverse_algebraic(default_type(in.real()), sigma)),
           R(model.inverse_algebraic(default_type(in.imag()), sigma)));
}

// Exact-unbiased inverse of the forward variance-stabilising transform,
//   mapping a stabilised-domain (group) mean to the bias-free underlying level.
// Applied only to the per-group DC term so that the magnitude noise-floor bias
//   is not re-introduced into the denoised output.
template <typename T>
typename std::enable_if<!is_complex<T>::value, T>::type
vst_inverse_unbiased(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return T(model.inverse_unbiased(default_type(in), sigma));
}
template <typename T>
typename std::enable_if<is_complex<T>::value, T>::type
vst_inverse_unbiased(const NoiseModel::Base &model, const T in, const default_type sigma) {
  using R = typename T::value_type;
  return T(R(model.inverse_unbiased(default_type(in.real()), sigma)),
           R(model.inverse_unbiased(default_type(in.imag()), sigma)));
}

// Local gain of the inverse transform at the operating point: the linear factor
//   by which a stabilised-domain residual is mapped back to the intensity scale.
// For complex (Gaussian) data the gain is identical across the real and imaginary
//   channels and independent of the operating point, so a single real factor suffices.
template <typename T>
typename std::enable_if<!is_complex<T>::value, default_type>::type
vst_jacobian(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return model.jacobian(default_type(in), sigma);
}
template <typename T>
typename std::enable_if<is_complex<T>::value, default_type>::type
vst_jacobian(const NoiseModel::Base &model, const T in, const default_type sigma) {
  return model.jacobian(default_type(in.real()), sigma);
}

} // namespace MR::Denoise::Precondition
