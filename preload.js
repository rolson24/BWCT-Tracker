const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),
  saveCountsFile: () => ipcRenderer.send('save-counts-file'),
  saveProcessedVideoFile: () => ipcRenderer.send('save-processed-video-file'),
  saveLineCrossingsFile: () => ipcRenderer.send('save-line-crossings-file'),

});

