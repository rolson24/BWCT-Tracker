const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const { powerMonitor } = require('electron');
const path = require('path');
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
let mainWindow;
let flaskProcess = null;
const fs = require('fs');
const fsExtra = require('fs-extra'); // Import fs-extra

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1024,
        height: 768,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        }
    });

    // It's a good practice to also handle window close events
    mainWindow.on('close', function() {
        mainWindow.webContents.executeJavaScript('localStorage.clear();');
    });
}

// Function to check if the backend is responsive and reload if necessary
function checkBackendHealth() {
    fetch('http://localhost:5000/health')
        .then(response => {
            if (!response.ok) {
                // If the response is not ok, reload the window
                console.log('Backend is not responsive, reloading window...');
                mainWindow.reload();
            }
        })
        .catch(error => {
            console.error('Error checking backend health:', error);
            // If there is an error, reload the window
            mainWindow.reload();
        });
}

// Set an interval to check the backend health every 5 minutes
setInterval(checkBackendHealth, 300000); // 300000 milliseconds = 5 minutes

function checkBackendConnectionAndReconnect() {
    // Example: Attempt to fetch a simple endpoint from your backend
    fetch('http://localhost:5000/health')
        .then(response => {
            if (response.ok) {
                console.log('Backend connection is healthy.');
                // Reload your Electron app's renderer process, if necessary
                mainWindow.reload();
            } else {
                console.log('Backend connection failed. Trying again...');
                // Retry connection after a delay
                setTimeout(checkBackendConnectionAndReconnect, 5000); // Retry after 5 seconds
            }
        })
        .catch(error => {
            console.log('Error connecting to backend:', error);
            // Retry connection after a delay
            setTimeout(checkBackendConnectionAndReconnect, 5000); // Retry after 5 seconds
        });
}

async function getRawTracksFilePath() {
    try {
        // Removed the dynamic import as we've already imported node-fetch at the top
        const response = await fetch('http://localhost:5000/get_raw_tracks_file_path');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const countsFilePath = await response.text();
        return countsFilePath;
    } catch (error) {
        console.error("Failed to get tracks file path:", error);
        return null;
    }
}

async function getCountsFilePath() {
    try {
        // Removed the dynamic import as we've already imported node-fetch at the top
        const response = await fetch('http://localhost:5000/get_counts_file_path');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const countsFilePath = await response.text();
        return countsFilePath;
    } catch (error) {
        console.error("Failed to get counts file path:", error);
        return null;
    }
}

async function getProcessedVideoFilePath() {
    try {
        // Removed the dynamic import as we've already imported node-fetch at the top
        const response = await fetch('http://localhost:5000/get_processed_video_file_path');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const processedVideoFilePath = await response.text();
        return processedVideoFilePath;
    } catch (error) {
        console.error("Failed to get processed video file path:", error);
        return null;
    }
}

async function getCrossingsFilePath() {
    try {
        // Removed the dynamic import as we've already imported node-fetch at the top
        const response = await fetch('http://localhost:5000/get_line_crossings_file_path');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const crossingsFilePath = await response.text();
        return crossingsFilePath;
    } catch (error) {
        console.error("Failed to get crossings file path:", error);
        return null;
    }
}

async function getPlots(){
    try {
        // Removed the dynamic import as we've already imported node-fetch at the top
        const response = await fetch('http://localhost:5000/get_plots');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const plots_dir = await response.text();
        return plots_dir;
    } catch (error) {
        console.error("Failed to get plots data:", error);
        return null;
    }
}

function startFlaskApp() {
    flaskProcess = spawn('python', ['./backend/BWCT_app.py']);

    flaskProcess.stdout.on('data', (data) => {
    	console.log(`stdout: ${data}`);
    });


    flaskProcess.stderr.on('data', (data) => {
        console.log(`stderr: ${data}`);
        // This check might not work as expected because `data` is a Buffer, not a string
        // Consider converting `data` to a string before checking its contents
        if (data.toString().includes("Running on http://")) {
            console.log("Flask is ready. Loading window...");
            mainWindow.loadURL('http://127.0.0.1:5000');

         // Listen for an IPC message to open the file dialog
            ipcMain.handle('open-file-dialog', async () => {
                const { filePaths } = await dialog.showOpenDialog({
                    properties: ['openFile']
                });
                console.log(`initial filenames: ${filePaths}`)
                return filePaths;
            });

            ipcMain.on('save-raw-tracks-file', async (event) => {
                const tracksFilePath = await getRawTracksFilePath();  // Retrieve the counts file path

                if (!tracksFilePath || tracksFilePath.includes("No video file provided") || tracksFilePath.includes("Tracks file not found") || tracksFilePath.includes("No runs found")) {
                    console.error('Failed to get raw tracks file path:', tracksFilePath);
                    event.sender.send('save-raw-tracks-file-response', 'failure');
                    return;
                }
                const { canceled, filePath } = await dialog.showSaveDialog({
                    title: 'Save Tracks File',
                    buttonLabel: 'Save',
                    // Suggest a default file name if you like
                    defaultPath: path.join(app.getPath('downloads'), 'raw_data.zip'),
                    // Set filters to limit to specific file types (optional)
                    filters: [
                        { name: 'Zip Files', extensions: ['zip'] },
                    ]
                });

                if (canceled || !filePath) {
                    // User cancelled the dialog or closed it without selecting a location
                    return;
                }

                // Copy the file to the user-selected location
                fs.copyFile(tracksFilePath, filePath, (err) => {
                    if (err) {
                        console.error('Failed to save tracks file:', err);
                        event.sender.send('save-raw-tracks-file-response', 'failure');
                    } else {
                        console.log('Tracks file saved successfully');
                        event.sender.send('save-raw-tracks-file-response', 'success');
                    }
                });
            });

            ipcMain.on('save-counts-file', async (event) => {
                const countsFilePath = await getCountsFilePath();  // Retrieve the counts file path

                if (!countsFilePath || countsFilePath.includes("No video file provided") || countsFilePath.includes("Counts file not found") || countsFilePath.includes("No runs found")) {
                    console.error('Failed to get counts file path:', countsFilePath);
                    event.sender.send('save-counts-file-response', 'failure');
                    return;
                }
                const { canceled, filePath } = await dialog.showSaveDialog({
                    title: 'Save Counts File',
                    buttonLabel: 'Save',
                    // Suggest a default file name if you like
                    defaultPath: path.join(app.getPath('downloads'), 'counts_output.txt'),
                    // Set filters to limit to specific file types (optional)
                    filters: [
                        { name: 'Text Files', extensions: ['txt'] },
                    ]
                });

                if (canceled || !filePath) {
                    // User cancelled the dialog or closed it without selecting a location
                    return;
                }

                // Copy the file to the user-selected location
                fs.copyFile(countsFilePath, filePath, (err) => {
                    if (err) {
                        console.error('Failed to save counts file:', err);
                        event.sender.send('save-counts-file-response', 'failure');
                    } else {
                        console.log('Counts file saved successfully');
                        event.sender.send('save-counts-file-response', 'success');
                    }
                });


            });

            ipcMain.on('save-processed-video-file', async (event) => {
                const videoFilePath = await getProcessedVideoFilePath();  // Retrieve the counts file path

                if (!videoFilePath || videoFilePath.includes("No video file provided") || videoFilePath.includes("Video file not found") || videoFilePath.includes("No runs found")) {
                    console.error('Failed to get video file path:', videoFilePath);
                    event.sender.send('save-processed-video-file-response', 'failure');
                    return;
                }
                const { canceled, filePath } = await dialog.showSaveDialog({
                    title: 'Save Processed Video File',
                    buttonLabel: 'Save',
                    // Suggest a default file name if you like
                    defaultPath: path.join(app.getPath('downloads'), 'processed_video.mp4'),
                    filters: [
                        { name: 'MP4 Video', extensions: ['mp4'] }
                    ],
                });

                if (canceled || !filePath) {
                    // User cancelled the dialog or closed it without selecting a location
                    return;
                }

                // Copy the file to the user-selected location
                fs.copyFile(videoFilePath, filePath, (err) => {
                    if (err) {
                        console.error('Failed to save processed video file:', err);
                        event.sender.send('save-processed-video-file-response', 'failure');
                    } else {
                        console.log('Processed video file saved successfully');
                        event.sender.send('save-processed-video-file-response', 'success');
                    }
                });


            });

            ipcMain.on('save-line-crossings-file', async (event) => {
                const crossingsFilePath = await getCrossingsFilePath();  // Retrieve the counts file path

                if (!crossingsFilePath || crossingsFilePath.includes("No video file provided") || crossingsFilePath.includes("Counts file not found") || crossingsFilePath.includes("No runs found")) {
                    console.error('Failed to get line crossings file path:', crossingsFilePath);
                    event.sender.send('save-line-crossings-file-response', 'failure');
                    return;
                }
                const { canceled, filePath } = await dialog.showSaveDialog({
                    title: 'Save Line Crossings File',
                    buttonLabel: 'Save',
                    // Suggest a default file name if you like
                    defaultPath: path.join(app.getPath('downloads'), 'line_crossings.txt'),
                    // Set filters to limit to specific file types (optional)
                    filters: [
                        { name: 'Text Files', extensions: ['txt'] },
                    ]
                });

                if (canceled || !filePath) {
                    // User cancelled the dialog or closed it without selecting a location
                    return;
                }

                // Copy the file to the user-selected location
                fs.copyFile(crossingsFilePath, filePath, (err) => {
                    if (err) {
                        console.error('Failed to save line crossings file:', err);
                        event.sender.send('save-line-crossings-file-response', 'failure');
                    } else {
                        console.log('Line crossings file saved successfully');
                        event.sender.send('save-line-crossings-file-response', 'success');
                    }
                });
            });

            ipcMain.on('save-plots-folder', async (event) => {
                const plotsDirPath = await getPlots();  // Retrieve the counts file path

                if (!plotsDirPath || plotsDirPath.includes("No video file provided") || plotsDirPath.includes("Counts file not found") || plotsDirPath.includes("No runs found")) {
                    console.error('Failed to get plots directory path:', plotsDirPath);
                    event.sender.send('save-plots-folder-response', 'failure');
                    return;
                }
                const { canceled, filePaths } = await dialog.showOpenDialog({
                    title: 'Select Destination Folder for Plots',
                    buttonLabel: 'Select', // This label might not always be customizable for directory dialogs
                    properties: ['openDirectory', 'createDirectory']
                });

                if (canceled || filePaths.length === 0) {
                    console.log('No folder selected.');
                    event.sender.send('save-plots-folder-response', 'canceled');
                    return;
                }

                const destinationPath = filePaths[0]; // The selected directory path
                console.log("Selected folder:", destinationPath);

                // The new directory where files will be copied
                const newPlotsDir = path.join(destinationPath, 'plots');

                // Create the new directory and copy files
                fsExtra.ensureDir(newPlotsDir, (err) => {
                    if (err) {
                        console.error('Failed to create plots directory:', err);
                        event.sender.send('save-plots-folder-response', 'failure');
                        return;
                    }
                    // Copy the directory
                    fsExtra.copy(plotsDirPath, newPlotsDir, (copyErr) => {
                        if (copyErr) {
                            console.error('Failed to copy plots directory:', copyErr);
                            event.sender.send('save-plots-folder-response', 'failure');
                        } else {
                            console.log('Plots directory copied successfully');
                            event.sender.send('save-plots-folder-response', 'success');
                        }
                    });
                });
            });
        }
    });


}

app.on('ready', () => {
    startFlaskApp();
    createWindow();
    // Simulate the computer going to sleep after 3 minutes
    // setTimeout(() => {
    //     powerMonitor.emit('suspend');
    //     // Simulate the computer waking up after 3 more minutes
    //     setTimeout(() => {
    //         powerMonitor.emit('resume');
    //     }, 120000); // 180000 milliseconds = 3 minutes
    // }, 120000); // 180000 milliseconds = 3 minutes
    powerMonitor.on('suspend', () => {
        console.log('The system is going to sleep');
    });

    powerMonitor.on('resume', () => {
        console.log('The system has resumed from sleep');
        checkBackendConnectionAndReconnect();
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});



app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});

app.on('before-quit', (event) => {
    // This is now the primary place to handle cleanup
    if (flaskProcess) {
        console.log('Shutting down Flask process...');
        flaskProcess.kill('SIGINT'); // Send SIGINT to mimic Ctrl+C

        setTimeout(() => {
            console.log('Force quitting Electron app...');
            app.exit(0);
        }, 5000);
    }
    // No need to call app.quit() here; letting the event proceed naturally
});

// Remove the 'will-quit' event handler to simplify the quit process
