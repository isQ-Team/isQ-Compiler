#include <iostream>
#include <cstring>
using namespace std;

int main(int argc, char* argv[])
{
    if (argc < 3) return -1;
    size_t pathlen = strlen(argv[1]);
    if (argv[1][pathlen - 1] == '/')
        cout << argv[1] << argv[2];
    else
        cout << argv[1] << '/' << argv[2];
    return 0;
}