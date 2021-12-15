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

std::vector<std::string> links;
std::map<std::string, std::set<size_t>> index1;
size_t link_num = 0;

void read_file(std::string path) {
    ifstream f;
    f.open (path, ios::in | ios::binary);
    f.seekg (0, ios::end);
    size_t length = f.tellg();
    f.seekg (0, ios::beg);
    char sym;
    std::string s;
    for(size_t i = 0; i < length; ++i) {
        f.read(&sym, sizeof(sym));
        s += sym;
    }
    s += "    ";
    std::string link;
    std::string text;
    size_t i = 0;
    while(s[i] != 'h' || s[i + 1] != 't' || s[i + 2] != 't' || s[i + 3] != 'p') {
        i++;
    }
    while(i < length) {
        if(i < length && s[i] == 'h' && s[i + 1] == 't' && s[i + 2] == 't' && s[i + 3] == 'p') {
            link = "";
            while(int(s[i]) != 26 && s[i] != ' ') {
                link += s[i];
                i++;
            }
        } else {
            text = "";
            while(i < length && (s[i] != 'h' || s[i + 1] != 't' || s[i + 2] != 't' || s[i + 3] != 'p')) {
                text += s[i];
                i++;
            }
            ToLower(text);
            links.push_back(link);
            istringstream ss(text);
            string word;
            while (ss >> word) {
                if(index1.find(word) == index1.end()) {
                    index1[word] = std::set<size_t>();
                }
                index1[word].insert(link_num);
            }
            link_num++;
        }
    }
    f.close();
}

int main() 
{
    for(char c = '1'; c <= '8'; ++c) {
        std::string path = "./dataset/";
        path += c;
        read_file(path);
    }
    // ofstream out("index");
    // for(auto it = index1.begin(); it != index1.end(); ++it) {
    //     out << it->first << " ";
    //     out << index1[it->first].size() << " ";
    //     for(auto i : index1[it->first]) {
    //         out << i << " ";
    //     }
    //     out << std::endl;
    // }
    // ofstream out1("links");
    // std::cout << "links[0]" << links[0] << std::endl;
    // for(auto i : links) {
    //     out1 << i;
    //     out1 << std::endl;
    // }
    // out.close();
    // out1.close();
    cout << "Index created" << std::endl;

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
            result = index1[word];
        } else {
            std::set<size_t> res;
            for(auto i : result) {
                if(std::binary_search(index1[word].begin(), index1[word].end(), i)) {
                    res.insert(i);
                }
            }
            result = res;
        }
    }
    cout << "Amount of found links: " << result.size() << std::endl;
    for(auto i : result) {
        cout << links[i] << std::endl;
    }
}