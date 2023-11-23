/*
 * main-table.cpp
 * 
 * Copyright 2023 Jan Nikl <j.nikl@hzdr.de>
 * 
 */

#include "EOSTableSimple.h"
#include "Array.h"
#include "Parser.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <iomanip> // For setting the width of the output
#include <string>

using namespace std;

class Entry {
	public:
	string fileName;
	Array1 <string> element;
	Array1 <double> number;

	string GetName() const {
		if (element.Size()==1) {
			return NameFromChemicalSymbol(element[0],true);
		} else {
			string name;
			for(int i=0; i<element.Size(); i++) {
	name += element[i];
	if (fabs(number[i]-1.0)>1e-10) {
		if (fabs(number[i]-round(number[i]))<1e-10) {
			name += IntToString(int(number[i]));
		} else {
			name += DoubleToString(number[i]);
		}
	}
			}
			return name;
		}
	}

	bool IsAnElement() const {
		return (element.Size()==1);
	}
};

class Database {
	public:
	Array1 <Entry> entries;

	void ReadDataBaseFile(const string & fileName) {
		Parser p(fileName);
		p.SetIgnoreComments();
		p.SetIgnoreEmptyLines();
		while (p.ReadLine()) {
			Entry entry;
			entry.fileName = p.GetString(0);
			for(int i=1; i<p.GetNWords(); i+=2) {
	entry.element.PushBack(p.GetString(i));
	entry.number.PushBack(p.GetDouble(i+1));
			}
			entries.PushBack(entry);
		}
		cout << "Parsed database file \"" << fileName << "\" with " << entries.Size() << " entries." << endl;
	}

	EOSTableDiffT ReadInEOSTable(const int k) const {
		double mFU = 0.0;
		for(int i=0; i<entries[k].element.Size(); i++) {
			mFU += entries[k].number[i] * MassNumberFromName(entries[k].element[i]);
		}
		mFU *= PC::u/PC::me; // convert from nuclear mass unit 'u' to electron masses (atomic units)
		double nAtomsPerFormulaUnit = entries[k].number.Sum();
		EOSTableDiffT eos;
		eos.ReadTableInStandardFormat(entries[k].fileName,nAtomsPerFormulaUnit,mFU);
		eos.SetChargesFormulaUnit(entries[k].element,entries[k].number);
		eos.ExtractHugoniotDataFromFile(entries[k].fileName);
		eos.ExtractReferencesFromFile(entries[k].fileName);
		eos.SetName(entries[k].GetName());
		eos.SetDatabaseIndex(k);
		return eos;
	}

	void Print() const {
		for(int i=0; i<entries.Size(); i++) {
			cout << IntToStringMaxNumber(i+1,entries.Size(),' ');
			cout << ": ";
			cout << entries[i].GetName();
			cout << endl;
		}
	}

	void PrintInfoForAllEOS() const {
		Array1 <EOSTableDiffT> eos;
		for(int i=0; i<entries.Size(); i++) { // separate reading part from print the info
			eos.PushBack(ReadInEOSTable(i));
		}
		for(int i=0; i<entries.Size(); i++) {
			eos[i].PrintInfo();
		}
	}

	int Size() const {
		return entries.Size(); 
	}

};

// int main(int argc, char **argv)
// {
// 	Database database;
// 	database.ReadDataBaseFile("fpeos_database.txt");

// 	const int iEOS = 10; //10 = Al
// 	EOSTableDiffT eos = database.ReadInEOSTable(iEOS-1);

// 	vector<double> rhos({ 1., 1.27, 1.62, 2.07, 2.64, 3.36, 4.28, 5.46, 6.95, 8.86, 11.29, 14.38, 18.33, 23.36, 29.76, 37.93, 48.33, 61.58, 78.48, 100.});
// 	vector<double> tmps({3.16, 3.89, 4.77, 5.87, 7.21, 8.86, 10.89, 13.38, 16.44, 20.2, 24.82, 30.49, 37.47, 46.04, 56.58, 69.52, 85.42, 104.97, 128.98, 158.49});

// 	const double rho_min = eos.MassDensityGCC(eos.GetMinimumDensity());
// 	const double rho_max = eos.MassDensityGCC(eos.GetMaximumDensity());

// 	cout << "density [g/cc]\ttemperature [eV]\tpressure [GPa]" << endl;

// 	for(double rho : rhos)
// 	{
// 		double nnn = eos.NumberDensityFromMassDensityGCC(rho);
		
// 		if(rho < rho_min || rho > rho_max)
// 			continue;

// 		const double tmp_min = eos.GetMinimumTemperature(nnn);
// 		const double tmp_max = eos.GetMaximumTemperature(nnn);

// 		for(double tmp: tmps)
// 		{
// 			double tmpAU = tmp * PC::eVToAU;

// 			if(tmpAU < tmp_min || tmpAU > tmp_max)
// 				continue;
			
// 			double p = eos.Pressure(nnn, tmpAU) * PC::AUToGPa;
// 			cout << rho << "\t" << tmp << "\t" << p << endl;
// 		}
// 	}
	
// 	return 0;
// }

// Assuming the other classes and constants are defined elsewhere in your code

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <ElementCode> <iEOS>" << std::endl;
        return 1;
    }

    std::string elementCode = argv[1];
    std::string splitnum = argv[2];
    std::unordered_map<std::string, int> element_to_iEOS;

    // Insert key-value pairs into the map
    element_to_iEOS["H"] = 1;
    element_to_iEOS["He"] = 2;
    element_to_iEOS["B"] = 3;
    element_to_iEOS["C"] = 4;
    element_to_iEOS["N"] = 5;
    element_to_iEOS["O"] = 6;
    element_to_iEOS["Ne"] = 7;
    element_to_iEOS["Na"] = 8;
    element_to_iEOS["Mg"] = 9;
    element_to_iEOS["Al"] = 10;
    element_to_iEOS["Si"] = 11;            
    int iEOS = element_to_iEOS[elementCode];

    // Construct file names
    std::string inputFileName = elementCode + "_EOS_test_input_"+splitnum+".txt";
    std::string inputModFileName = elementCode + "_EOS_test_input_x_"+splitnum+".txt";
    std::string outputFileName = elementCode + "_EOS_test_output_x_"+splitnum+".txt";
    Database database;  // Create a Database object
    // std::string DataBaseFileName = "fpeos_database_"+splitnum+".txt";
    database.ReadDataBaseFile("fpeos_database_"+splitnum+".txt"); // Read data from the file

    EOSTableDiffT eos = database.ReadInEOSTable(iEOS-1); // Read in EOS table

    const double rho_min = eos.MassDensityGCC(eos.GetMinimumDensity()); // Get minimum mass density
    const double rho_max = eos.MassDensityGCC(eos.GetMaximumDensity()); // Get maximum mass density

    // Create an input and output files
    std::ifstream infile(inputFileName);
    std::ofstream outfile(outputFileName);
    std::ofstream infilemod(inputModFileName);

    // Print header line
    outfile << "rho\ttemp\tpressure" << std::endl;
    infilemod << "rho\ttemp\tpressure" << std::endl;

    std::string line;

    // Skip the header line
    std::getline(infile, line);

    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        double rho, tmp, p_inp;
        iss >> rho >> tmp >> p_inp;    

        double nnn = eos.NumberDensityFromMassDensityGCC(rho); // Convert mass density to number density

        // Check if the value of rho is within the valid range, and continue to the next iteration if not
        // if(rho < rho_min || rho > rho_max)
        //     continue;

        // Get the minimum and maximum valid temperatures for this density
        const double tmp_min = eos.GetMinimumTemperature(nnn);
        const double tmp_max = eos.GetMaximumTemperature(nnn);

        double tmpAU = tmp * PC::eVToAU; // Convert temperature to atomic units

        // Check if the value of tmpAU is within the valid range, and continue to the next iteration if not
        // if(tmpAU < tmp_min || tmpAU > tmp_max)
        //     continue;

        // Calculate pressure and convert to GPa
        double p = eos.Pressure(nnn, tmpAU) * PC::AUToGPa;

	// Writing the aligned data to the file
	outfile << std::setw(12) << rho << "\t" << std::setw(12) << tmp << "\t" << std::setw(12) << p << std::endl;
	infilemod << std::setw(12) << rho << "\t" << std::setw(12) << tmp << "\t" << std::setw(12) << p_inp << std::endl;
    }

    infile.close(); // Close the file

    return 0; // Return success
}


