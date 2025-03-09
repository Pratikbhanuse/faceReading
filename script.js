const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const processedImage = document.getElementById("processedImage");

// Access webcam
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error("Webcam access error:", err));

function sendFrameToServer() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append("image", blob);

        fetch("http://127.0.0.1:5000/video_feed", {
            method: "POST",
            body: formData
        })
        .then(response => response.blob())
        .then(blob => {
            processedImage.src = URL.createObjectURL(blob);
        })
        .catch(err => console.error("Server request error:", err));
    }, "image/jpeg");
}

setInterval(sendFrameToServer, 1000);
