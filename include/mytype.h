#ifndef __MYTYPE_H_
#define __MYTYPE_H_

#include <string>
#include <complex>

namespace mytype {
template<typename T>
std::string type_str() { return ""; }
template<> inline
std::string type_str<int>() { return "i"; }
template<> inline
std::string type_str<long>() { return "l"; }
template<> inline
std::string type_str<double>() { return "d"; }
template<> inline
std::string type_str<std::complex<double>>() { return "cd"; }

}

#endif // __MYTYPE_H_