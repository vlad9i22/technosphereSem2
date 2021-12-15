#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <ios>
#include <regex>
#include <set>

using namespace std;
class Varbyte {
public:
    Varbyte() {
    }

    void EncodeNumber(int n) {
        std::vector <unsigned char> number;
        std::stack<unsigned char> stack;
        while(n) {
            stack.push(n % 128);
            n = n / 128;
        }
        while(stack.size()) {
            if(stack.size() == 1) {
                number.push_back(stack.top() | 128);
            } else {
                number.push_back(stack.top());
            }
            stack.pop();
        }
        for (auto i : number) {
            s.push_back(i);
        }
    }
    std::set<size_t> decode_sequence(std::vector<unsigned char> dec) {
        std::set<size_t> res;
        size_t current_num = 0;
        size_t num = 0;
        for(auto i : dec) {
            if(i < 128) {
                current_num = (current_num << 7) + i;
            } else {
                current_num = (current_num << 7) + i - 128;
                num += current_num;
                res.insert(num);
                current_num = 0;
            }
        }
        return res;
    }
    std::vector<unsigned char> s;
};
int main() 
{
    Varbyte encoder;
    unsigned char a = 255;
    encoder.EncodeNumber(123534);
    encoder.EncodeNumber(12335);
    encoder.EncodeNumber(2534);
    encoder.EncodeNumber(3534);
    for(auto i : encoder.decode_sequence(encoder.s)) {
        cout << i << std::endl;
    }
}