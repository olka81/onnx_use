#include "text_generator.h"

TextGenerator::TextGenerator() {
    digit_to_joke = {
        {0, L"Zero is the only number that can't be represented in Roman numerals."},
        {1, L"One is the loneliest number."},
        {2, L"Two's company, three's a crowd."},
        {3, L"Three is a magic number."},
        {4, L"Four-tunately, you're not alone!"},
        {5, L"High five!"},
        {6, L"Six of one, half a dozen of the other."},
        {7, L"Seven days without laughter makes one weak."},
        {8, L"Eight ate nine."},
        {9, L"Cloud 9 is where the fun begins."}
    };
}

std::wstring TextGenerator::generate(int digit) const 
{
    auto it = digit_to_joke.find(digit);
    return (it != digit_to_joke.end()) ? it->second :  L"Unknown digit.";  //cpp 20 => use .contains
}
