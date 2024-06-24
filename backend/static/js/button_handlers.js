window.handleImportVideoButton = function handleImportVideoButton() {
    $('#video-upload-button').on('click', function(e) {
        // Prevent the default form submission if the button is part of a form
        e.preventDefault();
        console.log("Open video button")

        // Call the openFileDialog function exposed by the preload script
        window.electronAPI.openFileDialog().then(result => {
            // Assuming openFileDialog is adjusted to return a Promise that resolves with the file selection result
            console.log("get filenames")
            console.log(result)
            // You can now do something with the selected file paths, like sending them to your Flask backend
            $.ajax({
                url: '/receive-file-paths',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ filenames: result }),
                success: function(response) {
                    console.log('Server response:', response);
                    $('#draw-button').show();
                    $('#process-button').show();  // Show the "Process Video" button
                    $('#reprocess-button').show();  // Show the "Reprocess Counts" button
                    $('#clear-lines-button').show();  // Show the "Clear Lines" button
                    // var video_name = response.filename.split('.')[0];
                    // var video_filename = response.filename;

                    $('#video-player').attr('src', `/stream_video`, 'type', 'video/mp4').css({ width: '100%', height: '100%' }).prop('controls', true).on('loadedmetadata', function () {
                        var canvas = $('#canvas')[0];
                        var ctx = canvas.getContext('2d');
                        video_width = this.videoWidth;
                        video_height = this.videoHeight;
                        var aspectRatio = this.videoHeight / this.videoWidth;
                        var canvasHeight = this.clientWidth * aspectRatio;
                        console.log(`Canvas height: ${canvasHeight}`)
                        var yOffset = (this.clientHeight - canvasHeight) / 2
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        $('#canvas').css("top", yOffset).attr({ width: this.clientWidth, height: canvasHeight }).css('pointer-events', 'none').show();
                    }).show();

                    $('#reprocess-button').prop('disabled', true)

                    $('#download-counts-link').attr('href', '/download_counts');
                    $('#download-processed-video-link').attr('href', '/download_processed_video');
                },
                error: function(xhr, status, error) {
                    console.error('Error sending filenames:', error);
                }
            })
        }).catch(err => {
            console.error('Error opening file dialog:', err);
        });
    });
}

function getBasename(filePath) {
    // Use a regular expression to extract the part after the last slash
    return filePath.replace(/^.*[\\\/]/, '');
}
const getMeta = (url) =>
    new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
        img.src = url;
    });

window.handleImportRawDataButton = function handleImportRawDataButton() {
    $('#raw-data-button').on('click', function(e) {
        // Prevent the default form submission if the button is part of a form
        e.preventDefault();

        // Call the openFileDialog function exposed by the preload script
        window.electronAPI.openFileDialog().then(result => {
            // Assuming openFileDialog is adjusted to return a Promise that resolves with the file selection result
            console.log("get filenames")
            console.log(result)
            // You can now do something with the selected file paths, like sending them to your Flask backend
            $.ajax({
                url: '/receive-raw-tracks-file-path',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ filenames: result }),
                success: function(response) {
                    console.log('Server response:', response);
                    $('#draw-button').show();
                    $('#process-button').show();  // Show the "Process Video" button
                    $('#reprocess-button').show();  // Show the "Reprocess Counts" button
                    $('#clear-lines-button').show();  // Show the "Clear Lines" button
                    // var video_name = response.filename.split('.')[0];
                    // var video_filename = response.filename;

                    video_name = getBasename(result[0]).split('.')[0];
                    image_name = `/static/outputs/${video_name}/run_1/middle_frame.jpg`
                    console.log(`image name: ${image_name}`)

                    $('#video-player').css({ width: '100%', height: '100%' }).prop('controls', false).show();
                    var videoPlayer = document.getElementById("video-player")


                    ;(async() => {
                        const img = await getMeta(image_name);
                        console.dir(img.naturalHeight + ' ' + img.naturalWidth);
                        video_width = img.naturalWidth;
                        video_height = img.naturalHeight;

                        var canvas = $('#canvas')[0];
                        var ctx = canvas.getContext('2d');
                        var aspectRatio = video_height / video_width;
                        console.log(`Video Aspect Ratios: ${aspectRatio}`)
                        var canvasHeight = videoPlayer.clientWidth * aspectRatio;
                        console.log(`Canvas height: ${canvasHeight}`)
                        var yOffset = (videoPlayer.clientHeight - canvasHeight) / 2
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        $('#canvas').css("top", yOffset).attr({ width: videoPlayer.clientWidth, height: canvasHeight }).css('pointer-events', 'none').show();
                        console.log("setup canvas")
                    })();


                    // Use the Fetch API to make a request to the file's URL
                    fetch(image_name)
                        .then(response => {
                            // Check if the file is accessible
                            if(response.ok) {
                                console.log('The file is accessible.');
                                videoPlayer.poster = image_name
                            } else {
                                console.log('The file is not accessible.', response.statusText);
                                // Handle the case where the file is not accessible
                            }
                        })
                        .catch(error => {
                            // Handle any errors that occur during the fetch
                            console.error('Error fetching the file:', error);
                        });



                    $('#process-button').prop('disabled', true)

                    $('#download-counts-link').attr('href', '/download_counts');
                    $('#download-processed-video-link').attr('href', '/download_processed_video');
                },
                error: function(xhr, status, error) {
                    console.error('Error sending filenames:', error);
                }
            })
        }).catch(err => {
            console.error('Error opening file dialog:', err);
        });
    });
}


window.handleClearLinesButton = function handleClearLinesButton() {
    $('#clear-lines-button').on('click', function () {
        var ctx = $('#canvas')[0].getContext('2d');
        ctx.clearRect(0, 0, $('#canvas')[0].width, $('#canvas')[0].height);
        $.post('/clear_lines', function (response) {
        });
        for (var label in labels) {
            label_obj = labels[label];
            if (Array.isArray(label_obj)) {
                for (var dir_label in label_obj) {
                    console.log(`dir_label: ${label_obj[dir_label]} type: ${typeof label_obj[dir_label]}`)
                    label_obj[dir_label].remove()
                }
            } else {
                label_obj.remove()
            }
        }
        labels = {};
        labels['Out'] = [];
        labels['In'] = [];
    });
}

window.handleDrawButton = function handleDrawButton() {
    $('#draw-button').on('click', function () {
        console.log('Draw button clicked');
        drawing = !drawing;
        $('#video-player').prop('controls', !drawing);
        $('#canvas').css('pointer-events', drawing ? 'auto' : 'none');
        $(this).text(drawing ? 'Stop Drawing' : 'Draw Lines');
    });
}

window.handleProcessButton = function handleProcessButton() {
    $('#process-button').on('click', function () {
        if (processing == false) {
            var confirmProcess = confirm("Are you sure you want to process the full video? This could take several hours. Remember if you just want to process again with new lines, you can use the 'Reprocess Counts' button");
            if (!confirmProcess) {
                return; // User clicked 'No', exit the function
            }

            $('#process-button').prop('disabled', true)
            $('#reprocess-button').prop('disabled', true)

            processing = true
            $('#processing-progress').css('width', 0 + '%').attr('aria-valuenow', 0);
            var saveVideo = $('#save-video').val(); // Get the dropdown value

            $('#processing-loader').show()
            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',  // Indicate that you're sending JSON data
                },
                body: JSON.stringify({ save_video: saveVideo })  // Convert your data into a JSON string

            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Video processing started:', data);
                })
                .catch(error => {
                    console.error('There has been a problem with your fetch operation:', error);
                });
        }
    });
}

window.handleReprocessButton = function handleReprocessButton() {
    $('#reprocess-button').on('click', function () {
        $('#processing-progress').css('width', '0%').attr('aria-valuenow', 0);  // Reset the progress bar
        $('#processing-loader').show()

        fetch('/reprocess', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                console.log('Video reprocessing started:', data);
            })
            .catch(error => {
                console.error('There has been a problem with your fetch operation:', error);
            });
    });
}

window.handleDownloadButtons = function handleDownloadButtons() {
    document.getElementById('download-counts-button').addEventListener('click', () => {
        window.electronAPI.saveCountsFile();
    });

    $('#download-raw-data-button').on('click', function () {
        window.electronAPI.saveRawTracksFile();
    });

    $('#download-lines-button').on('click', function () {
        window.electronAPI.saveLineCrossingsFile();
    });

    $('#download-processed-video-button').on('click', function () {
        window.electronAPI.saveProcessedVideoFile();

    });

    $('#download-plots-button').on('click', function () {
        window.electronAPI.savePlots();
        $('#download-plots-button').prop('disabled', true);

    });
}