const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),
  saveRawTracksFile: () => ipcRenderer.send('save-raw-tracks-file'),
  saveCountsFile: () => ipcRenderer.send('save-counts-file'),
  saveProcessedVideoFile: () => ipcRenderer.send('save-processed-video-file'),
  saveLineCrossingsFile: () => ipcRenderer.send('save-line-crossings-file'),
  savePlots: () => ipcRenderer.send('save-plots-folder'),
});

