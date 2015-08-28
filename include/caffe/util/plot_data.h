//  Created by Aidan Gomez on 2015-08-12.
//  Copyright (c) 2015 Venture Media. All rights reserved.

#pragma once
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>

namespace caffe {

template <typename It>
void drawPNG(std:: string filename, It b, It e) {
    std::ofstream buf(filename, std::ios_base::out | std::ios_base::trunc);

    using value_type = typename std::iterator_traits<It>::value_type;
    auto stream = std::ostream_iterator<value_type>(buf, "\n");
    std::copy(b, e, stream);
    buf.close();

    auto cmd = "/usr/bin/gnuplot -e \"file=\'" + filename + "\'\" data/scripts/plot_data.gplot";
    std::system(cmd.c_str());

    std::remove(filename.c_str());
}

template <typename Container>
void drawPNG(std::string filename, Container& cont) {
    using namespace std;
    auto b = begin(cont);
    auto e = end(cont);

    return drawPNG(filename, b, e);
}

} // namespace caffe
