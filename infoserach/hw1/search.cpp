#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <ios>
#include <regex>
#include <set>
#include <stack>

using namespace std;

void ToLower(std::string &s) {
    for(size_t i = 0; i < s.size() - 1; ++i) {
        if(s[i] == char(0xd0) && s[i + 1] >= char(0x90) && s[i + 1] <= char(0xaf)) {
            if(s[i + 1] <= char(0x9f)) {
                s[i + 1] = char(0xb0) + s[i + 1] - char(0x90);
            } else {
                s[i + 1] = char(0x80) + s[i + 1] - char(0xa0);
                s[i] = char(0xd1);
            }
        } else {
            s[i] = std::tolower(s[i]);
        }
    }
}

int main() {
    std::vector<std::string> linksr;
    std::map<std::string, std::set<size_t>> indexr;

    ifstream in("./index", ios::in | ios::binary);
    size_t cur_size;
    std::string s;
    while(in) {
        in >> s;
        in >> cur_size;
        size_t i = 0;
        size_t l_num;
        while(i < cur_size) {
            in >> l_num;
            if(indexr.find(s) == indexr.end()) {
                indexr[s] = std::set<size_t>();
            }
            indexr[s].insert(l_num);
            i++;
        }
    }
    ifstream in1("./links", ios::in);
    while(in1) {
        in1 >> s;
        linksr.push_back(s);
    }
    in.close();
    in1.close();

    std::string queue;
    std::cin >> queue;
    ToLower(queue);
    std::set<size_t> result;
    string word;
    istringstream ss(queue);
    bool flag = 0;
    while (ss >> word) {
        if(!flag) {
            flag = 1;
            result = indexr[word];
        } else {
            std::set<size_t> res;
            for(auto i : result) {
                if(std::binary_search(indexr[word].begin(), indexr[word].end(), i)) {
                    res.insert(i);
                }
            }
            result = res;
        }
    }
    cout << "Amount of found links: " << result.size() << std::endl;
    for(auto i : result) {
        cout << linksr[i] << std::endl;
    }
}