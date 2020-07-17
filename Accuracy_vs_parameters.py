import matplotlib.pyplot as plt

plt.close('all')
plt.figure(figsize=(6, 4))
#plt.plot([10000, 102034, 234553], [70, 80, 90], '.-', label='Convolutional models')

plt.plot([110, 238, 438], [91.1, 95.8, 95.2], '.-', markersize=12, label='res_original')
plt.plot([19.9, 42.6, 78.4], [90.1, 94, 93.3], '.-',markersize=12, label='res_narrow_original')
plt.plot([23, 53, 92], [95.5, 96.4, 96.6], '.-',markersize=12, label='res_narrow_improved')
plt.plot([57], [96.1], '.-', markersize=12, label='res8_lite')
plt.plot([202], [96.9], '.-', markersize=12, label='Att_original')
plt.plot([25, 50, 87, 155], [96.6, 96.8, 97, 97.1], '.-', markersize=12, label='Att_improved')


# plt.semilogy(val_loss, label='Validation loss')
plt.title('Accuracy vs Number of Parameters', pad=10, fontsize=15)
plt.xlabel('Number of parameters (in thousands)')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("acc_vs_param.pdf")
plt.show()