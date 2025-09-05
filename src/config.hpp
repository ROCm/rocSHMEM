/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_CONFIG_HPP_
#define LIBRARY_SRC_CONFIG_HPP_

#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

namespace rocshmem {
namespace config {
  namespace category {
    // env var categories
    // when adding a new category, make sure to add prefix<tag::CATEGORY>
    enum class tag {
      ROCSHMEM,
      BOOTSTRAP,
      REVERSE_OFFLOAD,
    };

    // env var string prefixes
    // prevent instantiation of default template; require specializations for each tag
    // if P2041 (see https://wg21.link/P2041) gets merged, can be changed to just = delete instead
    template <tag C> inline constexpr std::enable_if_t<!std::is_enum_v<decltype(C)>> prefix;
    template <> inline constexpr const char* prefix<tag::ROCSHMEM> = "ROCSHMEM";
    template <> inline constexpr const char* prefix<tag::BOOTSTRAP> = "ROCSHMEM_BOOTSTRAP";
    template <> inline constexpr const char* prefix<tag::REVERSE_OFFLOAD> = "ROCSHMEM_RO";
  }  // namespace category

  namespace parser {
    // base parser template
    // calls operator>>(std::istream&, T&)
    template <typename T>
    struct parse {
      std::istream& operator()(std::istream& is, T& value) const {
        // accept all bases for integer types
        if constexpr (std::is_integral_v<T>) {
          is >> std::setbase(0);
        }
        return is >> value;
      }
    };

    // string parser specialization, parse entire line
    // operator>>(std::istream&, std::string&) stops on the first whitespace character
    template <> inline
    std::istream& parse<std::string>::operator()(std::istream& is, std::string& value) const {
      return std::getline(is, value);
    }

    // bool parser specialization, parse both false/true and 0/1
    // to accept true/false, True/False, on/off, On/Off, ON/OFF, 0/1, etc.,
    // can create a facet inheriting from std::num_get<char> and overriding do_get(..., bool& v)
    // then use the locale with that facet: is.imbue(std::locale(is.getloc(), new bool_get{}))
    // note that std::locale is responsible for reference-counting the facets, which is very silly
    // can the locale (and/or facets) have static storage duration?
    template <> inline
    std::istream& parse<bool>::operator()(std::istream& is, bool& value) const {
      auto pos = is.tellg();
      is >> std::boolalpha >> value;
      if (is.fail()) {
        is.clear();
        is.seekg(pos);
        is >> std::noboolalpha >> value;
      }
      return is;
    }

    // decimal integer parser
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    struct parse_decimal {
      std::istream& operator()(std::istream& is, T& value) const {
        return is >> std::dec >> value;
      }
    };

    // hexadecimal integer parser
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    struct parse_hex {
      std::istream& operator()(std::istream& is, T& value) const {
        return is >> std::hex >> value;
      }
    };
  }  // namespace parser

  // class var<Type, Category>
  // reads the specified environment variable using std::getenv()
  // if it set, the variable is parsed (using parser::parse<Type> by default)
  // if it is unset or parsing fails, a default value is used instead
  template <typename T, category::tag C = category::tag::ROCSHMEM>
  class var {
  public:
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
    static constexpr category::tag category = C;

    // primary constructor
    template <typename Parser>
    var(const std::string& _name, const std::string& _doc, const_reference _default_value,
        Parser parse);

    // convenience (delegating) constructors
    template <typename Parser>
    var(const std::string& _name, const std::string& _doc, Parser parse)
      : var(_name, _doc, T{}, parse) { }
    var(const std::string& _name, const std::string& _doc, const_reference _default_value)
      : var(_name, _doc, _default_value, parser::parse<T>{}) { }
    var(const std::string& _name, const std::string& _doc)
      : var(_name, _doc, T{}, parser::parse<T>{}) { }

    // public accessors
    const std::string& get_name() const {
      return name;
    }
    const std::string& get_doc() const {
      return doc;
    }
    const_reference get_default() const {
      return default_value;
    }
    const_reference get_value() const {
      return value;
    }
    operator const_reference() const {
      return value;
    }

    // can't figure out how to do an out-of-line definition for this
    template <typename CharT, typename Traits>
    friend
    std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
                                                  const var<value_type, category>& v) {
      return os << v.name << "=" << v.value;
    }

  private:
    const std::string name;
    const std::string doc;
    const value_type default_value;
    value_type value;
  };

  template <typename T, category::tag C>
  template <typename Parser>
  var<T, C>::var(const std::string& _name, const std::string& _doc, const_reference _default_value,
                 Parser parse)
      : name(category::prefix<C> + _name),
        doc(_doc),
        default_value(_default_value),
        value(_default_value) {
    const char* env_value = std::getenv(name.c_str());
    if (env_value) {
      std::istringstream iss{std::string(env_value)};
      std::invoke(parse, iss, value);
      if (iss.fail()) {
        std::cerr << name << ": invalid argument '" << env_value << "'" << std::endl;
        value = default_value;
      }
    }
  }
}  // namespace config
}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONFIG_HPP_
