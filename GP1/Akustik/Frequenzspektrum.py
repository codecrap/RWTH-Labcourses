import PraktLib as pl
import matplotlib.pyplot as plt




    
    n, t, U_1 = pl.readLabFile("Daten_Akustik"/"gitarre"/"schwebung.lab")
    
    fig,ax = plt.subplots()
    
    ax.plot(t, U_1)
    
    freq, amp = pl.fourier_fft(t, U_1)
    fig, axis = plt.subplots()
    axis.semilogy(freq[freq>0], amp[freq>0])
    
    fig.savefig("Dropbox"/"ProtokollAkustik/"schwebung.lab.eps", format = "eps", dpi=256)
plt.show()






