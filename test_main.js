const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
let mainWindow;
let flaskProcess = null;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        }
    });

    // It's a good practice to also handle window close events
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
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
        }
    });
    

}

app.on('ready', () => {
    startFlaskApp();
    createWindow();
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

