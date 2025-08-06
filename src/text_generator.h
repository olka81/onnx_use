#pragma once
#include <string>
#include <map>

class TextGenerator {
public:
    TextGenerator();
    std::wstring generate(int digit) const;
private:
    std::map<int, std::wstring> digit_to_joke; //TODO: model
};