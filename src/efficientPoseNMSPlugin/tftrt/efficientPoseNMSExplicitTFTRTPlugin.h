/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_H
#define TRT_EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_H

#include <vector>

#include "common/plugin.h"
#include "efficientPoseNMSPlugin/efficientPoseNMSParameters.h"
#include "efficientPoseNMSPlugin/efficientPoseNMSPlugin.h"

// This plugin provides CombinedNMS op compatibility for TF-TRT in Explicit Batch
// and Dymamic Shape modes

namespace nvinfer1
{
namespace plugin
{

class EfficientPoseNMSExplicitTFTRTPlugin : public EfficientPoseNMSPlugin
{
public:
    explicit EfficientPoseNMSExplicitTFTRTPlugin(EfficientPoseNMSParameters param);
    EfficientPoseNMSExplicitTFTRTPlugin(const void* data, size_t length);
    ~EfficientPoseNMSExplicitTFTRTPlugin() override = default;
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    IPluginV2DynamicExt* clone() const noexcept override;
};

// TF-TRT CombinedNMS Op Compatibility
class EfficientPoseNMSExplicitTFTRTPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    EfficientPoseNMSExplicitTFTRTPluginCreator();
    ~EfficientPoseNMSExplicitTFTRTPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        const char* name, const void* serialData, size_t serialLength) noexcept override;

protected:
    PluginFieldCollection mFC;
    EfficientPoseNMSParameters mParam;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_H
