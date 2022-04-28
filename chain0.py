from DigiCommPy import modem
import numpy as np
import math
import matplotlib.pyplot as plt
from decBin import bin2Dec, dec2Bin
import numpy.matlib as mb
from utils import power
import time
from commpy import channelcoding as cd
#import commpy

#import numpy, scipy.io

# Constants
PI = np.pi

# Functions


# Variables
nOfdm = 5
nFft = 1024
mQam = 256
nCp = 32
nGuardBand = 256
lengthWord = 204
lengthCodeword = 408
nSamples = 8192
fOffset = 0.001
theta = fOffset*nFft*180
print(f'theta = {theta} degrees')
#nQamSymbols = (nOfdm)*nFft
nPayload = nFft-nGuardBand
nQamSymbols = (nOfdm)*nPayload
nBitsPerSymbol = int(math.log2(mQam))
nBits = int(nQamSymbols*nBitsPerSymbol)
nDataSamples = (nFft+nCp)*(nOfdm+2)
nZeroPading = nSamples-nDataSamples
if nDataSamples>nSamples:
    raise NameError('Too many samples')
ldpc_code_params = cd.get_ldpc_code_params('408.33.844.txt')
nCodeWords = nBits//lengthCodeword
nDataBits = nCodeWords*lengthWord
nCodedBits = nCodeWords*lengthCodeword
nPading = nBits % lengthCodeword
# snr = 20
#Blocks
qamModem = modem.QAMModem (mQam)

timeStart = time.perf_counter()
nItterations = 1


def create_pilots(mQam, nFft):
    pilotDec0 = np.random.randint(mQam, size=(1, nFft))
    pilotDec1 = np.random.randint(mQam, size=(1, nFft))
    qamPilot0 = qamModem.modulate(pilotDec0)       # Modulate the pilot
    qamPilot1 = qamModem.modulate(pilotDec1)       # Modulate the pilot
    ofdmPilot0Half = np.fft.ifft(qamPilot0, int(nFft/2))*math.sqrt(nFft/2)  # ifft
    ofdmPilot0 = np.concatenate((ofdmPilot0Half, ofdmPilot0Half), axis=1)
    ofdmPilot1 = np.fft.ifft(qamPilot1, nFft)*math.sqrt(nFft)  # ifft
    cpPilot0 = np.concatenate((ofdmPilot0[:, nFft-nCp:nFft], ofdmPilot0), axis=1)   # add CP
    cpPilot1 = np.concatenate((ofdmPilot1[:, nFft-nCp:nFft], ofdmPilot1), axis=1)   # add CP
    return cpPilot0, cpPilot1, ofdmPilot0, ofdmPilot1, qamPilot1


def channel_encode(bitStream, ldpc_code_params, nPading):
    codeWords = cd.triang_ldpc_systematic_encode(bitStream, ldpc_code_params, pad=True)
    encodedStream = np.reshape(np.array(codeWords), -1)
    paddedStream = np.concatenate((encodedStream, np.ones(nPading, dtype=int)))
    return paddedStream


def qam_mapping(paddedStream, nQamSymbols, nBitsPerSymbol):
    DecStream = bin2Dec(paddedStream, nQamSymbols, nBitsPerSymbol)  # convert to decimal (bit grouping)
    qamSymbols = qamModem.modulate(DecStream)  # Modulate the bit stream
    return qamSymbols


def ofdm_modulate(qamSymbols, nPayload, nOfdm, nGuardBand, nFft, nCp):
    parallelSymbolsPayload = qamSymbols.reshape(-1, nPayload)  # serial to parallel
    parallelSymbols = np.concatenate((parallelSymbolsPayload[:, 0:int(nPayload / 2)], np.zeros((nOfdm, nGuardBand)),
                                      parallelSymbolsPayload[:, int(nPayload / 2):nPayload]), axis=1)
    ofdmSymbols = np.fft.ifft(parallelSymbols, nFft) * math.sqrt(nFft)  # ifft
    cpSymbols = np.concatenate((ofdmSymbols[:, nFft - nCp:nFft], ofdmSymbols), axis=1)  # add CP
    serialSymbols = cpSymbols.reshape(1, -1)  # parallel to serial
    return serialSymbols


def physical_frame(serialSymbols, cpPilot0, cpPilot1, nZeroPading):
    signalOut = np.concatenate((cpPilot0, cpPilot1, serialSymbols), axis=1)
    signalOut = signalOut / math.sqrt(power(signalOut))
    signal = np.concatenate((np.zeros((1, int(nZeroPading / 2))), signalOut, np.zeros((1, int(nZeroPading / 2)))),
                            axis=1)
    return signal


def time_freq_sync(recSignal, cpPilot0, ofdmPilot0, ofdmPilot1, nFft, nCp, nDataSamples):
    # detect the start
    xcorRes = np.correlate(recSignal.transpose(), np.squeeze(cpPilot0), mode='full')
    peakIndex = np.argmax(abs(xcorRes))
    # start = peakIndex-cpPilot0.size+1
    # extractedSignal = recSignal[start:start+nDataSamples]

    # Inital Frequency offset correction: offset < 1 subcarrier (<PI)
    r = recSignal
    L = int(nFft / 2)
    P = np.zeros(r.size - nFft) + 1j * np.zeros(r.size - nFft)
    R = np.zeros(r.size - nFft)
    for d in range(0, r.size - nFft):
        P[d] = np.sum(np.conjugate(r[d:d + L]) * r[d + L:d + 2 * L])
        R[d] = np.sum(abs(r[d + L:d + 2 * L]) ** 2)
    X = P / R
    X[np.argwhere(abs(X) > 1)] = 0
    maxX = max(abs(X))
    maxXIndex = np.argmax(abs(X))
    x = np.argwhere(abs(X) > (0.995 * maxX))
    start = int(x[0])
    phy0 = np.angle(P[x[0]])
    fOffsetEst = phy0 / nFft / PI
    # if fOffsetEst<1e-6:
    #    fOffsetEst = 0
    print(f'index xcorr = {peakIndex - cpPilot0.size + 1}, index SC = {int(x[0])}')
    extractedSignal = recSignal[start:start + nDataSamples]
    extractedSignalCorrected = extractedSignal * np.exp(-1j * 2 * PI * fOffsetEst * tLine[0:extractedSignal.size])

    # Extract pilots to use in second step of ofset correction
    recPilot0 = extractedSignalCorrected[0:nFft + nCp]
    recPilot1 = extractedSignalCorrected[nFft + nCp:2 * (nFft + nCp)]
    recPilot0NoCp = recPilot0[nCp:]
    recPilot1NoCp = recPilot1[nCp:]
    recPilotFreq0 = np.fft.fft(recPilot0NoCp, nFft) / math.sqrt(nFft)  # FFT
    recPilotFreq1 = np.fft.fft(recPilot1NoCp, nFft) / math.sqrt(nFft)  # FFT
    x1k = recPilotFreq0
    x2k = recPilotFreq1

    c0 = np.squeeze(np.fft.fft(ofdmPilot0, nFft) / math.sqrt(nFft))
    c1 = np.squeeze(np.fft.fft(ofdmPilot1, nFft) / math.sqrt(nFft))
    # vk = c1/c0
    # ve2 = vk[0:nFft:2]
    ve2 = math.sqrt(2) * np.squeeze(c1[0:nFft:2] / c0[0:nFft:2])
    xe1 = x1k[0:nFft:2]
    xe2 = x2k[0:nFft:2]

    # secondary offset correction: integer ofset (multiple of subcarriers)
    qIndex = 0
    qRange = np.arange(-L / 2, L / 2)
    G = np.zeros(L) + 1j * np.zeros(L)

    for q in qRange:
        xs1 = np.roll(xe1, -int(q))
        xs2 = np.roll(xe2, -int(q))
        G[qIndex] = np.sum(np.conjugate(xs1) * np.conjugate(ve2) * xs2)
        qIndex += 1

    Den = 2 * (np.sum(abs(xe2) ** 2)) ** 2
    B = abs(G) ** 2 / Den
    gEst0 = np.argmax(abs(B))
    print(f'gEst0 = {gEst0}')
    gEst = gEst0 - nFft / 4
    f0FinalEst = np.squeeze(fOffsetEst) + gEst * 2 / nFft
    print(f'the estimated offset = {f0FinalEst} while the real offset = {fOffset}')

    # correct the received signal
    extractedSignalCorrected = extractedSignal * np.exp(-1j * 2 * PI * f0FinalEst * tLine[0:extractedSignal.size])
    recPilot1 = extractedSignalCorrected[nFft + nCp:2 * (nFft + nCp)]
    recPilot1NoCp = recPilot1[nCp:]
    recPilotFreq1 = np.fft.fft(recPilot1NoCp, nFft) / math.sqrt(nFft)  # FFT

    # Extract data
    recSymbols = extractedSignalCorrected[2 * (nFft + nCp):(nOfdm + 2) * (nFft + nCp)]

    return recSymbols,recPilotFreq1


def ofdm_demodulate(recSymbols, nOfdm, nFft, nCp):
    recPrllSymbols = recSymbols.reshape(nOfdm, (nFft + nCp))  # serial to parallel
    recNoCpSymbols = recPrllSymbols[:, nCp:]  # remove CP
    recPrllQamSymbols = np.fft.fft(recNoCpSymbols, nFft) / math.sqrt(nFft)  # FFT
    recSrlQamSymbols = recPrllQamSymbols.reshape(nOfdm * nFft, )  # serial to parallel
    return recSrlQamSymbols


def ofdm_equalization(recPilotFreq1, qamPilot1, nFft, nOfdm):
    # channel estimation:
    estChannelInitial = recPilotFreq1 / np.squeeze(qamPilot1.transpose())
    estChannelTime = np.fft.ifft(estChannelInitial) * math.sqrt(nFft)
    estChannelTime[10:nFft - 10] = np.zeros(nFft - 20)
    estChannel = np.fft.fft(estChannelTime, nFft) / math.sqrt(nFft)
    # Equalization
    equalizedSymbols = recSrlQamSymbols.reshape(nOfdm, nFft) / mb.repmat(estChannel, nOfdm, 1)
    return equalizedSymbols, estChannel


def qam_demapping(equalizedSymbols, nPayload, nGuardBand, nFft, nOfdm, nQamSymbols, nBitsPerSymbol):
    recPayload = np.concatenate(
        (equalizedSymbols[:, 0:int(nPayload / 2)], equalizedSymbols[:, nGuardBand + int(nPayload / 2):nFft]), axis=1)
    recQamSymbols = recPayload.reshape(-1, nPayload * nOfdm)  # parallel to serial
    recDec = qamModem.demodulate(np.squeeze(recQamSymbols))  # QAM demodulate
    recBits = dec2Bin(recDec, nQamSymbols, nBitsPerSymbol)  # get the binary stream
    return recBits

def error_correction(recBits, nCodedBits, lengthCodeword, lengthWord, nCodeWords, ldpc_code_params):
    recBitsNoPading = recBits[0:nCodedBits]
    recCodeWords = np.reshape(recBitsNoPading, (lengthCodeword, -1))
    correctedBitStream = np.zeros(nCodeWords * lengthWord, 'int')
    for k in range(nCodeWords):
        zInitial = np.array(cd.ldpc_bp_decode(recCodeWords[:, k], ldpc_code_params, 'MSA', 10))
        correctedBitStream[k * lengthWord:(k + 1) * lengthWord] = zInitial[1, 0:lengthWord]
    return correctedBitStream


snrList = [30]
ber = np.zeros((len(snrList), nItterations))
for m in range(len(snrList)):
    snr = snrList[m]
    for n in range(nItterations):
        # =============================================================================
        # Tx
        # =============================================================================
        
        # Pilot preparation:
        cpPilot0, cpPilot1, ofdmPilot0, ofdmPilot1, qamPilot1 = create_pilots(mQam, nFft)
        # generate the data:
        bitStream = np.random.randint(2, size=nDataBits)   # generate a random bit stream
        # channel encoding:
        paddedStream = channel_encode(bitStream, ldpc_code_params, nPading)
        # modulate
        qamSymbols = qam_mapping(paddedStream, nQamSymbols, nBitsPerSymbol)
        # OFDM modulation
        serialSymbols = ofdm_modulate(qamSymbols, nPayload, nOfdm, nGuardBand, nFft, nCp)
        # add pilots
        signal = physical_frame(serialSymbols, cpPilot0, cpPilot1, nZeroPading)
        # plot the spectrum
        # pSignal = power(signal)
        # Signal = np.squeeze(np.fft.fft(signal)/math.sqrt(np.squeeze(signal.size)))
        # plt.figure(0)
        # ax = plt.semilogy(20*np.log10(np.fft.fftshift(abs(Signal))))
        # plt.grid()
        # plt.show()

        print(f'bitStream: size of bitStream is {bitStream.size}')
        print(f'channel_encode: size of output is {paddedStream.size}')
        print(f'qam_mapping: size of output is {qamSymbols.size}')
        print(f'ofdm_modulate: size of output is {serialSymbols.size}')
        print(f'physical_frame: size of output is {signal.size}')

        # =============================================================================
        # channel
        # =============================================================================
        # channel impulse response (fading) preparation
        channelIR = (np.array([0, 0, 0, 1, -0.12, 0.054, -0.019, 0.0012])
                     + 1j*np.array([0, 0, 0, 1, -0.15, 0.057, -0.013, 0.0019]))/math.sqrt(2)
        # channelIR = np.array([1])
        G = np.fft.fft(channelIR, nFft)/math.sqrt(nFft)
        G = G/math.sqrt(power(G))
        # noise preparation
        powerSignal = power(signal)
        snrLin = 10**(snr/10)
        noisePowerSet = powerSignal/snrLin
        noise = math.sqrt(noisePowerSet)*(np.random.randn(signal.size) + 1j*np.random.randn(signal.size))/math.sqrt(2)
        powerNoise = power(noise)
        snrMeasured = 10*math.log10(powerSignal/powerNoise)
        # apply on the signal
        fadeSignal = np.convolve(np.squeeze(signal), channelIR.transpose(), mode='same') #filter the signal with the channel
        recSignal = fadeSignal+noise
        tLine = np.arange(recSignal.size)
        recSignal = recSignal*np.exp(1j*2*PI*fOffset*tLine)     # add the frequency ofset effect
        
        
        # =============================================================================
        # Rx
        # =============================================================================

        recSymbols, recPilotFreq1 = time_freq_sync(recSignal, cpPilot0, ofdmPilot0, ofdmPilot1, nFft, nCp, nDataSamples)
        recSrlQamSymbols = ofdm_demodulate(recSymbols, nOfdm, nFft, nCp)
        equalizedSymbols, estChannel = ofdm_equalization(recPilotFreq1, qamPilot1, nFft, nOfdm)
        recBits = qam_demapping(equalizedSymbols, nPayload, nGuardBand, nFft, nOfdm, nQamSymbols, nBitsPerSymbol)
        correctedBitStream = error_correction(recBits, nCodedBits, lengthCodeword, lengthWord, nCodeWords, ldpc_code_params)
        ber[m, n] = np.mean(abs(correctedBitStream-bitStream))

        print(f'channel: size of output is {recSignal.size}')
        print (f'time_freq_sync: size of output is {recSymbols.size}')
        print(f'ofdm_demodulate: size of output is {recSrlQamSymbols.size}')
        print(f'ofdm_demodulate: size of output is {recBits.size}')
        print(f'ofdm_demodulate: size of output is {correctedBitStream.size}')





        #print(f'Time needed for the itteraton = {time.perf_counter()-timeStart} ')
print(f'BER = {ber} for SNR = {snr}')
# =============================================================================
# BER = np.mean(ber,axis = 1)
# plt.figure(0)
# plt.semilogy(snrList,BER)
# plt.grid()
# =============================================================================

#scipy.io.savemat('arrdata.mat', mdict={'BER_16': BER})


# =============================================================================
# plt.figure(5)
# plt.plot(np.real(np.squeeze(recQamSymbols)),np.imag(np.squeeze(recQamSymbols)),'o')
# plt.grid()
# 
# plt.figure(6)
# plt.plot(np.squeeze(equalizedSymbols[3,:]))
# plt.plot(np.squeeze(parallelSymbols[3,:]))
# plt.figure(7)
# plt.plot(np.squeeze(recPayload[3,600:768]))
# plt.plot(np.squeeze(parallelSymbolsPayload[3,600:768]))
# E1 = np.mean(np.mean(abs(equalizedSymbols-parallelSymbols)))
# E2 = np.mean(np.mean(abs(recPayload-parallelSymbolsPayload)))
# E3 = np.mean(abs(recQamSymbols-qamSymbols))
# =============================================================================

