// Function to save the current state to local storage
window.saveStateToLocalStorage = function saveStateToLocalStorage(document, processing) {
    var video_player = document.getElementById('video-player');
    var state = {
        processing: processing,
        // lineStart: lineStart,
        videoPath: video_player.getAttribute('src'),
        saveVideoOption: document.getElementById('save-video').value,
        // Add other state variables you need to save here
    };
    console.log(`video src saved: ${state.videoPath}`)
    localStorage.setItem('appState', JSON.stringify(state));
}

// Function to load the state from local storage
window.loadStateFromLocalStorage = function loadStateFromLocalStorage(document) {
    var state = JSON.parse(localStorage.getItem('appState'));
    if (state) {
        processing = state.processing;
        // lineStart = state.lineStart;
        document.getElementById('video-player').attr('src', state.videoPath);
        document.getElementById('save-video').val(state.saveVideoOption);
        // Restore other parts of the state here
        // Update the UI to reflect the restored state
        console.log(`video src loaded: ${state.videoPath}`)
    }
}

// Function to save the canvas state to local storage
window.saveCanvasState = function saveCanvasState(canvas) {
    var canvasState = canvas.toDataURL(); // Save canvas drawing as data URL
    localStorage.setItem('canvasState', canvasState);
}

// Function to load the canvas state from local storage
window.loadCanvasState = function loadCanvasState(canvas) {
    var ctx = canvas.getContext('2d');
    var canvasState = localStorage.getItem('canvasState');
    if (canvasState) {
        var image = new Image();
        image.onload = function() {
            ctx.drawImage(image, 0, 0); // Draw the saved image onto the canvas
        };
        image.src = canvasState;
    }
}

// Function to save the counts data to local storage
window.saveCountsDataToLocalStorage = function saveCountsDataToLocalStorage(countsData) {
    localStorage.setItem('countsData', JSON.stringify(countsData));
}

// Function to load the counts data from local storage
window.loadCountsDataFromLocalStorage = function loadCountsDataFromLocalStorage() {
    var countsData = JSON.parse(localStorage.getItem('countsData'));
    return countsData;
}

// Function to save the plot data to local storage
window.savePlotDataToLocalStorage = function savePlotDataToLocalStorage(plotData, plotName) {
    localStorage.setItem(plotName, JSON.stringify(plotData));
}

// Function to load the plot data from local storage
window.loadPlotDataFromLocalStorage = function loadPlotDataFromLocalStorage(plotName) {
    var plotData = JSON.parse(localStorage.getItem(plotName));
    return plotData;
}

// Function to save the labels state to local storage
window.saveLabelsState = function saveLabelsState(labels) {
    var labelsState = [];
    var inLabelsState = [];
    var outLabelsState = [];
    for (var key in labels) {
        if (labels.hasOwnProperty(key)) {
            if (key === "In" || key === "Out") {
                labels[key].forEach(function(label) {
                    var labelData = {
                        text: label.textContent,
                        left: label.style.left,
                        top: label.style.top,
                        color: label.style.color
                    };
                    if (key === "In") {
                        inLabelsState.push(labelData);
                    } else {
                        outLabelsState.push(labelData);
                    }
                });
            } else {
                var label = labels[key];
                labelsState.push({
                    text: label.textContent,
                    left: label.style.left,
                    top: label.style.top,
                    color: label.style.color
                });
            }
        }
    }
    localStorage.setItem('labelsState', JSON.stringify(labelsState));
    localStorage.setItem('inLabelsState', JSON.stringify(inLabelsState));
    localStorage.setItem('outLabelsState', JSON.stringify(outLabelsState));
}

// Function to load the labels state from local storage and recreate labels
window.loadLabelsState = function loadLabelsState(document) {
    var labelsState = JSON.parse(localStorage.getItem('labelsState'));
    var inLabelsState = JSON.parse(localStorage.getItem('inLabelsState'));
    var outLabelsState = JSON.parse(localStorage.getItem('outLabelsState'));
    if (labelsState) {                                                               
        var videoContainer = document.querySelector('.video-container');             
        labelsState.forEach(function(labelState) {
        createLabel(labelState, "Line", document)
        });
    }                                                                
    if (inLabelsState) {                                                             
        inLabelsState.forEach(function(labelState) {                                 
            createLabel(labelState, "In", document);                                           
        });                                                                          
    }                                                                                
    if (outLabelsState) {                                                            
        outLabelsState.forEach(function(labelState) {                                
            createLabel(labelState, "Out", document);                                          
        });                                                                          
    } 
}

// Helper function to create a label                                                 
function createLabel(labelState, type, document) {                                             
    videoContainer = document.querySelector('.video-container');                 
    var label = document.createElement('span');                                      
    label.textContent = labelState.text;                                             
    label.style.position = 'absolute';                                               
    label.style.left = labelState.left;                                              
    label.style.top = labelState.top;                                                
    label.style.color = labelState.color;                                            
    label.style.fontSize = '15px';                                                   
    label.style.fontWeight = 'bold';                                                 
    label.style.pointerEvents = 'none';                                              
    videoContainer.appendChild(label);                                               
    if (type === "In") {                                                             
        labels["In"].push(label);                                                    
    } else if (type === "Out") {                                                     
        labels["Out"].push(label);                                                   
    } else {                                                                         
        labels[label.textContent] = label;                                           
    }                                                                                
} 

// Function to reload the video player with the current video source
function reloadVideoPlayer(videoPlayer) {
    if (videoPlayer) {
        var currentSrc = videoPlayer.getAttribute('src');
        videoPlayer.src = ''; // Reset the source to force reload
        videoPlayer.load(); // Load the video player without source to clear previous state
        videoPlayer.src = currentSrc; // Set the source back to the original path
        videoPlayer.load(); // Load the video player with the new source
        videoPlayer.show(); // Show the video player
    }
}

// Function to check if a video has been uploaded and reload the video player
window.checkAndReloadVideoPlayer = function checkAndReloadVideoPlayer(videoPlayer) {
    // Check if the video player has a source and if it's not empty
    if (videoPlayer && videoPlayer.getAttribute('src')) {
        console.log("reload the video player");
        reloadVideoPlayer(videoPlayer); // Call the reloadVideoPlayer function if a source is present
    }
}  