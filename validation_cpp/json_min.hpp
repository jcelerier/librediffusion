// Tiny read-only JSON parser, just enough for the harness to read the golden
// config.json / manifest.json (objects, arrays, strings, numbers, bool, null).
// Not a general-purpose parser; no unicode escapes beyond the common ones.
#pragma once

#include <cctype>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace jsonm
{

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value
{
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<ValuePtr> arr;
  std::map<std::string, ValuePtr> obj;

  bool is_null() const { return type == Type::Null; }
  double as_num() const { return num; }
  int as_int() const { return static_cast<int>(num); }
  bool as_bool() const { return b; }
  const std::string& as_str() const { return str; }

  const ValuePtr& at(const std::string& key) const
  {
    auto it = obj.find(key);
    if(it == obj.end())
      throw std::runtime_error("json: missing key " + key);
    return it->second;
  }
  bool has(const std::string& key) const { return obj.count(key) > 0; }
};

class Parser
{
public:
  explicit Parser(const std::string& s) : s_(s) {}

  ValuePtr parse()
  {
    skip_ws();
    auto v = parse_value();
    return v;
  }

private:
  const std::string& s_;
  size_t i_ = 0;

  void skip_ws()
  {
    while(i_ < s_.size() && std::isspace((unsigned char)s_[i_]))
      i_++;
  }
  char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }
  char get() { return s_[i_++]; }

  ValuePtr parse_value()
  {
    skip_ws();
    char c = peek();
    if(c == '{')
      return parse_object();
    if(c == '[')
      return parse_array();
    if(c == '"')
      return parse_string();
    if(c == 't' || c == 'f')
      return parse_bool();
    if(c == 'n')
      return parse_null();
    return parse_number();
  }

  ValuePtr parse_object()
  {
    auto v = std::make_shared<Value>();
    v->type = Value::Type::Object;
    get(); // {
    skip_ws();
    if(peek() == '}') { get(); return v; }
    while(true)
    {
      skip_ws();
      auto key = parse_string();
      skip_ws();
      if(get() != ':')
        throw std::runtime_error("json: expected ':'");
      v->obj[key->str] = parse_value();
      skip_ws();
      char c = get();
      if(c == ',')
        continue;
      if(c == '}')
        break;
      throw std::runtime_error("json: expected ',' or '}'");
    }
    return v;
  }

  ValuePtr parse_array()
  {
    auto v = std::make_shared<Value>();
    v->type = Value::Type::Array;
    get(); // [
    skip_ws();
    if(peek() == ']') { get(); return v; }
    while(true)
    {
      v->arr.push_back(parse_value());
      skip_ws();
      char c = get();
      if(c == ',')
        continue;
      if(c == ']')
        break;
      throw std::runtime_error("json: expected ',' or ']'");
    }
    return v;
  }

  ValuePtr parse_string()
  {
    auto v = std::make_shared<Value>();
    v->type = Value::Type::String;
    skip_ws();
    if(get() != '"')
      throw std::runtime_error("json: expected string");
    std::string out;
    while(true)
    {
      char c = get();
      if(c == '"')
        break;
      if(c == '\\')
      {
        char e = get();
        switch(e)
        {
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          case 'r': out += '\r'; break;
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          default: out += e; break;
        }
      }
      else
        out += c;
    }
    v->str = out;
    return v;
  }

  ValuePtr parse_bool()
  {
    auto v = std::make_shared<Value>();
    v->type = Value::Type::Bool;
    if(s_.compare(i_, 4, "true") == 0) { v->b = true; i_ += 4; }
    else if(s_.compare(i_, 5, "false") == 0) { v->b = false; i_ += 5; }
    else throw std::runtime_error("json: bad bool");
    return v;
  }

  ValuePtr parse_null()
  {
    auto v = std::make_shared<Value>();
    v->type = Value::Type::Null;
    if(s_.compare(i_, 4, "null") == 0) i_ += 4;
    else throw std::runtime_error("json: bad null");
    return v;
  }

  ValuePtr parse_number()
  {
    auto v = std::make_shared<Value>();
    v->type = Value::Type::Number;
    size_t start = i_;
    while(i_ < s_.size() && (std::isdigit((unsigned char)s_[i_]) || s_[i_] == '-'
                             || s_[i_] == '+' || s_[i_] == '.' || s_[i_] == 'e' || s_[i_] == 'E'))
      i_++;
    v->num = std::stod(s_.substr(start, i_ - start));
    return v;
  }
};

inline ValuePtr parse_file(const std::string& path)
{
  std::string content;
  {
    std::ifstream f(path, std::ios::binary);
    if(!f)
      throw std::runtime_error("json: cannot open " + path);
    content.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  }
  Parser p(content);
  return p.parse();
}

} // namespace jsonm
