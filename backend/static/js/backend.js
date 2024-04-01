// This is the backend for the app.

// Function to handle Socket.IO client reconnection
function handleSocketReconnection(socket) {
    // Attempt to reconnect with an exponential backoff strategy
    var reconnectAttempts = 0;
    var maxReconnectAttempts = 5;
    var reconnect = function() {
        if (reconnectAttempts < maxReconnectAttempts) {
            setTimeout(function() {
                if (!socket.connected) {
                    console.log('Attempting to reconnect to the WebSocket server...');
                    socket.connect();
                    reconnectAttempts++;
                    reconnect();
                }
            }, Math.pow(2, reconnectAttempts) * 1000); // Exponential backoff
        } else {
            console.log('Max reconnection attempts reached. Could not reconnect to the WebSocket server.');
        }
    };
    reconnect();
}


var processing = false;
var labels = {};  // Create an object to store the labels


function beforeUnloadListener(event) {
    event.returnValue = 'Are you sure you want to leave?';
}

// Save the state to local storage when the page is unloaded
window.addEventListener('beforeunload', function (e) {
    saveStateToLocalStorage(this.document, processing);
    // Save the labels state to local storage
    saveLabelsState(labels);
    // Save the canvas state to local storage
    saveCanvasState(this.document.getElementById('canvas'));
    // Rest of the beforeunload event...
});

window.addEventListener('beforeunload', beforeUnloadListener);

var drawing = false;

$(document).ready(function () {
    // Load state and plot data from local storage
    loadStateFromLocalStorage(document);
    var countsData = loadCountsDataFromLocalStorage();
    renderCountsTable(countsData, document);
    var countsPlotData = loadPlotDataFromLocalStorage('countsPlotData');
    renderPlot(countsPlotData, 'counts-plot', 'counts_by_line');
    var crossingsPlotData = loadPlotDataFromLocalStorage('crossingsPlotData');
    renderPlot(crossingsPlotData, 'crossings-plot', 'counts_per_hour');
    var trackPlotData = loadPlotDataFromLocalStorage('trackPlotData');
    renderPlot(trackPlotData, 'track-plot', 'tracks_overlay');
    var personVolumePlotData = loadPlotDataFromLocalStorage('personVolumePlotData');
    renderPlot(personVolumePlotData, 'person-volume-plot', 'person_volume');

    var lineStart = null;

    $('#video-player, #canvas').hide();

    // Call checkAndReloadVideoPlayer when the page is ready
    checkAndReloadVideoPlayer(document.getElementById('video-player'));

    // Load the labels state when the page is ready
    loadLabelsState(document);

    // Load the canvas state when the page is ready
    loadCanvasState(document.getElementById('canvas'));

    $('.custom-file-input').on("change", function () {
        var fileName = $(this).val().split("\\").pop();
        $(this).siblings('.custom-file-label').addClass("selected").html(fileName);
    });

    handleImportVideoButton();
    
    handleImportRawDataButton();

    handleDrawButton();

    handleClearLinesButton();

    handleProcessButton();

    handleReprocessButton();

    handleDownloadButtons();

    const forceReflow = (element) => {
        $('#video-upload-button').focus()
    };

    labels['Out'] = [];
    labels['In'] = [];
    $('#canvas').on('click', function (e) {
        console.log('drawing %b', drawing)
        if (drawing) {
            var rect = this.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
            lineEnd = { x: x, y: y };
            if (lineStart) {
                var ctx = this.getContext('2d');
                ctx.beginPath();
                ctx.moveTo(lineStart.x, lineStart.y);
                ctx.lineTo(x, y);
                ctx.strokeStyle = 'red';
                ctx.lineWidth = 5;
                ctx.stroke();
                var video = document.getElementById('video-player');
                var canvas = document.getElementById('canvas');
                var scaleX = video.videoWidth / canvas.width;
                var scaleY = video.videoHeight / canvas.height;

                console.log(`scaleX: ${scaleX}, scaleY: ${scaleY}`)
                console.log(`Line drawn: (${Math.round(lineStart.x * scaleX)},${Math.round(lineStart.y * scaleY)}) (${Math.round(x * scaleX)},${Math.round(y * scaleY)})`)
                $.post('/coordinates', { line: `(${Math.round(lineStart.x * scaleX)},${Math.round(lineStart.y * scaleY)}) (${Math.round(lineEnd.x * scaleX)},${Math.round(lineEnd.y * scaleY)})` }, function (response) {
                });

                current_labels = drawLabels(lineStart, lineEnd, document);

                console.log(current_labels)
                label = current_labels[0];
                labels[label.outerText] = label;  // Store the label in the labels object
                labels["Out"].push(current_labels[1]);
                labels["In"].push(current_labels[2]);
                lineStart = null;

                // Apply to the body or specific element needing refresh
                forceReflow($('#video-container'));
            } else {
                lineStart = { x: x, y: y };
            }
        }
        forceReflow($('#video-container'));


        document.getElementById("thisisfucked").clear
    });


    var socket = io.connect('http://' + document.domain + ':' + location.port);
    socket.on('disconnect', function() {
        console.log('Disconnected from WebSocket server.');
        handleSocketReconnection(socket);
    });
    socket.on('connect_error', function(error) {
        console.log('Connection error:', error);
        handleSocketReconnection(socket);
    });
    socket.on('progress', function (msg) {
        console.log('Processing progress:', msg);
        $('#processing-progress').css('width', msg.data + '%').attr('aria-valuenow', msg.data);
        $('#estimated-time').text('Estimated time remaining: ' + msg.time)
    });
    socket.on('plot-download-ready', function() {
        console.log('Ready to download plots');
        $('#download-plots-button').prop('disabled', false);

    })
    var video_name;
    // Listen for the video_processed event
    socket.on('video_processed', function (msg) {
        console.log('Video processed event received:', msg);  // Log the event data
        $('#processing-progress').css('width', '100%').attr('aria-valuenow', 100);
        $('#estimated-time').text('Estimated time remaining: 00:00:00')
        $('#processing-loader').hide()
        processing = false
        $('#process-button').prop('disabled', false);
        $('#reprocess-button').prop('disabled', false);

        // Make an AJAX request to get the counts data
        fetch('/get_counts')
            .then(response => response.json())
            // Inside the fetch success callback
            .then(data => {
                // Parse the counts data into an array of objects
                countsData = data.countsData;
                var currentLine = 0;
                var avgFPS;
                video_name = data.filename
                // Parse the JSON string into a JavaScript object
                var fig = JSON.parse(data.plot);
                console.log(`plotly figure ${fig.data}`);
                // Use Plotly to render the graph
                var countsPlotData = { data: fig.data, layout: fig.layout };
                renderPlot(countsPlotData, 'counts-plot', video_name + '_counts_by_line');
                saveCountsDataToLocalStorage(countsData); // Save the counts data to local storage
                savePlotDataToLocalStorage(countsPlotData, 'countsPlotData');

                // Get the table body element
                var tableBody = document.getElementById('counts-display').querySelector('tbody');

                // Clear the table body
                tableBody.innerHTML = '';

                // Insert each count as a row in the table
                countsData.forEach(count => {
                    var row = document.createElement('tr');
                    var lineCell = document.createElement('td');
                    var classCell = document.createElement('td');
                    var directionCell = document.createElement('td');
                    var countCell = document.createElement('td');

                    lineCell.textContent = count.line;
                    classCell.textContent = count.class;
                    directionCell.textContent = count.direction;
                    countCell.textContent = count.count;

                    var className = 'class-' + count.class.replace(/\s+/g, '-');
                    row.classList.add(className);

                    row.appendChild(lineCell);
                    row.appendChild(classCell);
                    row.appendChild(directionCell);
                    row.appendChild(countCell);
                    tableBody.appendChild(row);
                });

                // Insert the average FPS into the table
                if (avgFPS) {
                    var row = document.createElement('tr');
                    var cell = document.createElement('td');
                    cell.colSpan = 4;
                    cell.textContent = 'Average FPS: ' + avgFPS;
                    row.appendChild(cell);
                    tableBody.appendChild(row);
                }
                $('#download-raw-data-button').show(); // Show the "Download Raw Data" button
                $('#download-lines-button').show();  // Show the "Download Line Crossings" button
                $('#download-counts-button').show();  // Show the "Download Counts Output" button
                $('#download-processed-video-button').show();  // Show the "Download Processed Video" button
                $('#download-plots-button').show();  // Show the "Download Processed Video" button

            })
            .catch(error => {
                console.error('Error fetching counts data:', error);
            });
        console.log('get crossings data');

        var videoFPS = 30;  // Set the video FPS to 30
        fetch('/get_crossings_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // Include the fps in the body of the request
            body: JSON.stringify({ fps: videoFPS })
        })
            .then(response => response.json())
            .then(data => {
                var fig = JSON.parse(data.plot);
                // Use Plotly to render the graph
                var crossingsPlotData = { data: fig.data, layout: fig.layout };
                renderPlot(crossingsPlotData, 'crossings-plot', video_name + '_counts_per_hour');
                savePlotDataToLocalStorage(crossingsPlotData, 'crossingsPlotData');
            });

        fetch('/get_person_volume_data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            // Include the fps in the body of the request
            body: JSON.stringify({ fps: videoFPS })
        })
            .then(response => response.json())
            .then(data => {
                var fig = JSON.parse(data.plot);
                // Use Plotly to render the graph
                var personVolumePlotData = { data: fig.data, layout: fig.layout };
                renderPlot(personVolumePlotData, 'person-volume-plot', video_name + '_person_volume');
                savePlotDataToLocalStorage(personVolumePlotData, 'personVolumePlotData');
            });

        fetch('/get_tracks', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(response => response.json())
            .then(data => {
                // Parse the JSON string into a JavaScript object
                var fig = JSON.parse(data.fig_json);
                // Use Plotly to render the graph
                var trackPlotData = { data: fig.data, layout: fig.layout };
                renderPlot(trackPlotData, 'track-plot', video_name + '_tracks_overlay');
                savePlotDataToLocalStorage(trackPlotData, 'trackPlotData');

            })
            .catch(error => console.error('Error fetching the Plotly graph:', error));
    });
});