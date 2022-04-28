import numpy as np
import math
import matplotlib.pyplot as plt
import commpy
from commpy import channelcoding as cd

# Parameters:
nOfdm = 5
mQam = 64
nFft = 1024
nGuardBand = 256
nCP = 72
K = 204
N = 408
nDataBits = 204
snr = 25


# Coding and Modulation objects:
ldpc_code_params = cd.get_ldpc_code_params('408.33.844.txt')
Q = commpy.QAMModem(mQam)

# Tx:
bitStream = np.random.randint(2,size = nDataBits)   # generate a random bit stream
codeWords = cd.triang_ldpc_systematic_encode(bitStream, ldpc_code_params, pad=True)
encodedStream = np.reshape(np.array(codeWords),-1)
txSignal = Q.modulate(encodedStream)

# AWGN channel
powerSignal = np.mean(abs(txSignal)**2)
snrLin = 10**(snr/10)
noisePowerSet = powerSignal/snrLin
noise = math.sqrt(noisePowerSet) * (np.random.randn(txSignal.size) + 1j*np.random.randn(txSignal.size))   /  math.sqrt(2)
      
rxSignal = txSignal + noise
plt.figure(0)
plt.plot(np.real(rxSignal),np.imag(rxSignal),'o')
plt.plot(np.real(txSignal),np.imag(txSignal),'x')

# Rx:
demodSignalHard = Q.demodulate(rxSignal,'hard')
demodSignalSoft = Q.demodulate(rxSignal,'soft',noise_var=noisePowerSet)

recCodeWordsSoft = np.reshape(demodSignalSoft,(N,-1))
recCodeWordsHard = np.reshape(demodSignalHard,(N,-1))

finalSignalSoft = np.zeros(1*K,'int')
finalSignalHard = np.zeros(1*K,'int')
for n in range(1):
    zInitialSoft = np.array(cd.ldpc_bp_decode(recCodeWordsSoft[:,n], ldpc_code_params, 'MSA', 10))
    zInitialHard = np.array(cd.ldpc_bp_decode(recCodeWordsHard[:,n], ldpc_code_params, 'MSA', 10))
    finalSignalSoft[n*K:(n+1)*K] = zInitialSoft[0,0:K]
    finalSignalHard[n*K:(n+1)*K] = zInitialHard[1,0:K]

errSoft = np.mean(abs(finalSignalSoft-bitStream))
errHard = np.mean(abs(finalSignalHard-bitStream))

print(f'error Soft = {errSoft}')
print(f'error Hard = {errHard}')

