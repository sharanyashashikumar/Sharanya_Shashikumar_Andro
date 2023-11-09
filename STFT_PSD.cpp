#include <iostream>
#include <thread>
#include <vector>
#include <complex>
#include "fftw3.h"
#include <chrono>
#include <mutex>
#include "gnuplot-iostream.h"

const int signalLength = 1024;
const int fftSize = 256;
const int hopSize = 128;
const int bufferSize = 512;
const double sampleRate = 44100.0;

std::vector<std::complex<double>> inputSignal(signalLength); // Complex Input Signal
std::vector<std::vector<double>> stftBuffer(bufferSize, std::vector<double>(fftSize));
std::vector<double> psdBuffer(fftSize);

std::mutex stftMutex;
std::mutex psdMutex;

// Function to compute STFT
void computeSTFT() {
    for (int i = 0; i < bufferSize; i++) {
        std::vector<std::complex<double>> frame(inputSignal.begin() + i * hopSize, inputSignal.begin() + i * hopSize + fftSize);
        fftw_complex* in = reinterpret_cast<fftw_complex*>(&frame[0]);
        fftw_complex* out = reinterpret_cast<fftw_complex*>(&stftBuffer[i][0]);

        fftw_plan plan = fftw_plan_dft_1d(fftSize, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }
}
// Function to compute PSD
void computePSD() { 
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Update every 100 ms

        std::vector<std::complex<double>> frame(inputSignal.begin(), inputSignal.begin() + fftSize);
        fftw_complex* in = reinterpret_cast<fftw_complex*>(&frame[0]);
        fftw_complex* out = reinterpret_cast<fftw_complex*>(&psdBuffer[0]);

        fftw_plan plan = fftw_plan_dft_1d(fftSize, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);

        // Calculate the Power Spectrum Density (PSD)
        psdMutex.lock();
        for (int i = 0; i < fftSize; i++) {
            psdBuffer[i] = std::norm(psdBuffer[i]);
        }
        psdMutex.unlock();
    }
}

Gnuplot gp;

void displaySTFT() { // Displaying the graphs
    while (true) {
        computeSTFT(); /
        stftMutex.lock();

        gp << "set title 'STFT Plot'\n";
        gp << "set xlabel 'Time'\n";
        gp << "set ylabel 'Frequency'\n";
        gp << "splot '-' with lines notitle\n";

        for (int i = 0; i < bufferSize; i++) {
            for (int j = 0; j < fftSize; j++) {
                gp << i * hopSize << " " << j * sampleRate / fftSize << " " << stftBuffer[i][j] << "\n";
            }
        }

        gp << "e\n";
        gp.flush();
        stftMutex.unlock();
    }
}

void displayPSD() {
    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Update every 100 ms
        computePSD();
        // Display PSD data
        psdMutex.lock();

        gp << "set title 'PSD Plot'\n";
        gp << "set xlabel 'Frequency'\n";
        gp << "set ylabel 'Power'\n";
        gp << "plot '-' with lines notitle\n";

        for (int i = 0; i < fftSize; i++) {
            gp << i * sampleRate / fftSize << " " << psdBuffer[i] << "\n";
        }

        gp << "e\n";
        gp.flush();
        psdMutex.unlock();
    }
}

int main() {
   
    for (int i = 0; i < signalLength; i++) {
        inputSignal[i] = std::complex<double>(0.5 * cos(2 * M_PI * i / signalLength), 0.0);
    }

    std::thread stftThread(displaySTFT);
    std::thread psdThread(displayPSD);

    stftThread.join();
    psdThread.join();

    return 0;
}