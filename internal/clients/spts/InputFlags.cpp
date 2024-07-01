/********************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 ********************************************************************************/
#include <iomanip>
#include <vector>
#include <iostream>
#include "InputFlags.h"

InputFlags::InputFlags()
{
	AddInputFlag("help", 'h', "", "Print Help Message", "string");
}

void InputFlags::AddInputFlag(const std::string &_long_name,
							char _short_name,
							const std::string &_value,
							const std::string &_help_text,
							const std::string &_type)
{
	Input in;
	in.long_name = _long_name;
	in.short_name = _short_name;
	in.value = _value;
	in.help_text = _help_text;
	in.type = _type;

	if(MapInputs.count(_short_name) > 0)
		printf("Input flag: %s (%c) already exists !", _long_name.c_str(), _short_name);
	else
		MapInputs[_short_name] = in;
}

void InputFlags::Print()
{
	printf("SpTS Input Flags: \n\n");

	for(auto &content : MapInputs)
		std::cout<<std::setw(8)<<"--"<<content.second.long_name<<std::setw(20 - content.second.long_name.length())<<"-"<<content.first<<std::setw(8)<<" "<<content.second.help_text<<"\n";
	exit(0);
}

char InputFlags::FindShortName(const std::string &long_name)
{
	char short_name = '\0';

	for(auto &content : MapInputs)
	{
		if(content.second.long_name == long_name)
			short_name = content.first;
	}
	if(short_name == '\0')
	{
		std::cout<<"Long Name: "<<long_name<<" Not Found !";
		exit(0);
	}
	
	return short_name;
}

void InputFlags::Parse(int argc, char *argv[])
{
	std::vector<std::string> args;
	for(int i = 1; i < argc; i++)
		args.push_back(argv[i]);

	if(args.size() == 0) // No Input Flag
		Print();

	for(int i = 0; i < args.size(); i++)
	{
		std::string temp = args[i];
		if(temp[0] != '-')
		{
			printf("Illegal input flag\n");
			Print();
		}
		else if(temp[0] == '-' && temp[1] == '-') // Long Name Input
		{
			std::string long_name = temp.substr(2);
			if(long_name == "help")
				Print();

			char short_name = FindShortName(long_name);

            if (short_name == 'n' || short_name == 'z' || short_name == 'v')
            {
                MapInputs[short_name].value = "true";
            }
            else
            {
                MapInputs[short_name].value = args[i+1];
                i++;
            }
		}
		else if (temp[0] == '-' && temp[1] == '?') // Help Input
			Print();
		else // Short Name Input
		{
			char short_name = temp[1];
			if(MapInputs.find(short_name) == MapInputs.end())
			{
				std::cout<<"Input Flag: "<<short_name<<" Not Found !";
				exit(0);
			}
			if(short_name == 'h')
				Print();
            
            if(short_name == 'n' || short_name == 'z' || short_name == 'v' )
            {
                MapInputs[short_name].value = "true";
            }
            else
            {
                MapInputs[short_name].value = args[i+1];
                i++;
            }
		}
	}
}

std::string InputFlags::GetValueStr(const std::string &long_name)
{
	char short_name = FindShortName(long_name);
	std::string value = MapInputs[short_name].value;

	return value;
}	

int InputFlags::GetValueInt(const std::string &long_name)
{
	char short_name = FindShortName(long_name);
	int value = atoi(MapInputs[short_name].value.c_str());

	return value;
}

uint64_t InputFlags::GetValueUint64(const std::string &long_name)
{
    char short_name = FindShortName(long_name);
    uint64_t value = strtoull(MapInputs[short_name].value.c_str(), NULL, 10);

    return value;
}

float InputFlags::GetValueFloat(const std::string &long_name)
{
    char short_name = FindShortName(long_name);
    float value = std::stof(MapInputs[short_name].value);

    return value;

}

bool InputFlags::GetValueBool(const std::string &long_name)
{
    char short_name = FindShortName(long_name);
    if (MapInputs[short_name].value == "true")
        return true;
    else
        return false;
}
