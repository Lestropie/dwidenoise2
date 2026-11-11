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

#include "denoise/precondition/vst.h"

namespace MR::Denoise::Precondition {

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

template float vst_forward<float>(const NoiseModel::Base &, const float, const default_type);
template double vst_forward<double>(const NoiseModel::Base &, const double, const default_type);
template cfloat vst_forward<cfloat>(const NoiseModel::Base &, const cfloat, const default_type);
template cdouble vst_forward<cdouble>(const NoiseModel::Base &, const cdouble, const default_type);

template float vst_inverse<float>(const NoiseModel::Base &, const float, const default_type);
template double vst_inverse<double>(const NoiseModel::Base &, const double, const default_type);
template cfloat vst_inverse<cfloat>(const NoiseModel::Base &, const cfloat, const default_type);
template cdouble vst_inverse<cdouble>(const NoiseModel::Base &, const cdouble, const default_type);

template float vst_inverse_unbiased<float>(const NoiseModel::Base &, const float, const default_type);
template double vst_inverse_unbiased<double>(const NoiseModel::Base &, const double, const default_type);
template cfloat vst_inverse_unbiased<cfloat>(const NoiseModel::Base &, const cfloat, const default_type);
template cdouble vst_inverse_unbiased<cdouble>(const NoiseModel::Base &, const cdouble, const default_type);

template default_type vst_jacobian<float>(const NoiseModel::Base &, const float, const default_type);
template default_type vst_jacobian<double>(const NoiseModel::Base &, const double, const default_type);
template default_type vst_jacobian<cfloat>(const NoiseModel::Base &, const cfloat, const default_type);
template default_type vst_jacobian<cdouble>(const NoiseModel::Base &, const cdouble, const default_type);

} // namespace MR::Denoise::Precondition
