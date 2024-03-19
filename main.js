const { app, BrowserWindow } = require('electron')

const createWindow = () => {
  const win = new BrowserWindow({
    width: 800,
    height: 600, 
    webPreferences: {
    	nodeIntegration: true
    }
  })

  //win.loadFile('index.html')
  var python = require('child_process').spawn('py', ['./backend/webapp.py']);
  python.stdout.on('data', function (data) {
    console.log("data: ", data.toString('utf8'));
  });
  python.stderr.on('data', (data) => {
    console.log(`stderr: ${data}`); // when error
  });
}


app.whenReady().then(() => {
  createWindow()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on(‘activate’, () => {
 if (BrowserWindow.getAllWindows().length === 0) {
 createWindow()
 }
})
