// Copyright (c) 2017 Leosocy. All rights reserved.
// Use of this source code is governed by a MIT-style license 
// that can be found in the LICENSE file.

#include "core/palmprint.h"
#include "util/pub.h"

namespace edcc
{

Palmprint::Palmprint(const char *identity, const char *image_path)
    : identity_(identity),
    image_path_(image_path)
{
}

Palmprint::Palmprint(const Palmprint &rhs)
    : identity_(rhs.identity_),
    image_path_(rhs.image_path_),
    image_(rhs.image_.clone())
{
}

Palmprint& Palmprint::operator =(const Palmprint &rhs)
{
    if (this != &rhs)
    {
        identity_ = rhs.identity_;
        image_path_ = rhs.image_path_;
        image_ = rhs.image_.clone();
    }
    return *this;
}

bool Palmprint::operator==(const Palmprint &rhs) const
{
    return identity_ == rhs.identity_
        && image_path_ == rhs.image_path_;
}

cv::Mat* Palmprint::GetOrigImg()
{
//    image_ = cv::imread(image_path_, CV_LOAD_IMAGE_COLOR);
    image_ = cv::imread(image_path_);
    if (!image_.data)
    {
        EDCC_Log("Read image failed!");
        return NULL;
    }
    return &image_;
}

cv::Mat* Palmprint::GetSpecImg(const cv::Size &img_size, bool is_gray)
{
    Mat *origin_image = GetOrigImg();
    CHECK_POINTER_NULL_RETURN(origin_image, NULL);
    resize(*origin_image, *origin_image, img_size);
    if (is_gray)
    {
//        cvtColor(*origin_image, *origin_image, CV_BGR2GRAY);
        cvtColor(*origin_image, *origin_image, cv::COLOR_BGR2GRAY);
    }
    return origin_image;
}

} // namespace edcc
