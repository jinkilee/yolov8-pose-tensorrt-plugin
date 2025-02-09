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

#include "efficientPoseNMSExplicitTFTRTPlugin.h"
#include "efficientPoseNMSPlugin/efficientPoseNMSInference.h"

// This plugin provides CombinedNMS op compatibility for TF-TRT in Explicit Batch
// and Dymamic Shape modes

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using nvinfer1::plugin::EfficientPoseNMSExplicitTFTRTPlugin;
using nvinfer1::plugin::EfficientPoseNMSExplicitTFTRTPluginCreator;

namespace
{
const char* EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_VERSION{"1"};
const char* EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_NAME{"EfficientPoseNMS_Explicit_TF_TRT"};
} // namespace

EfficientPoseNMSExplicitTFTRTPlugin::EfficientPoseNMSExplicitTFTRTPlugin(EfficientPoseNMSParameters param)
    : EfficientPoseNMSPlugin(std::move(param))
{
}

EfficientPoseNMSExplicitTFTRTPlugin::EfficientPoseNMSExplicitTFTRTPlugin(const void* data, size_t length)
    : EfficientPoseNMSPlugin(data, length)
{
}

const char* EfficientPoseNMSExplicitTFTRTPlugin::getPluginType() const noexcept
{
    return EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_NAME;
}

const char* EfficientPoseNMSExplicitTFTRTPlugin::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_VERSION;
}

IPluginV2DynamicExt* EfficientPoseNMSExplicitTFTRTPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new EfficientPoseNMSExplicitTFTRTPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

EfficientPoseNMSExplicitTFTRTPluginCreator::EfficientPoseNMSExplicitTFTRTPluginCreator()
    : mParam{}
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("max_output_size_per_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_total_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("pad_per_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clip_boxes", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EfficientPoseNMSExplicitTFTRTPluginCreator::getPluginName() const noexcept
{
    return EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_NAME;
}

const char* EfficientPoseNMSExplicitTFTRTPluginCreator::getPluginVersion() const noexcept
{
    return EFFICIENT_NMS_EXPLICIT_TFTRT_PLUGIN_VERSION;
}

const PluginFieldCollection* EfficientPoseNMSExplicitTFTRTPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* EfficientPoseNMSExplicitTFTRTPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        const PluginField* fields = fc->fields;
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "max_output_size_per_class"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxesPerClass = *(static_cast<const int32_t*>(fields[i].data));
            }
            if (!strcmp(attrName, "max_total_size"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.numOutputBoxes = *(static_cast<const int32_t*>(fields[i].data));
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.iouThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_threshold"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                mParam.scoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            if (!strcmp(attrName, "pad_per_class"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.padOutputBoxesPerClass = *(static_cast<const int32_t*>(fields[i].data));
            }
            if (!strcmp(attrName, "clip_boxes"))
            {
                PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
                mParam.clipBoxes = *(static_cast<const int32_t*>(fields[i].data));
            }
        }

        auto* plugin = new EfficientPoseNMSExplicitTFTRTPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* EfficientPoseNMSExplicitTFTRTPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EfficientPoseNMSPlugin::destroy()
        auto* plugin = new EfficientPoseNMSExplicitTFTRTPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
