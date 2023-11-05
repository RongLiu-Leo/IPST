const slider = document.querySelector(".image-comparison .slider");
const beforeImage = document.querySelector(".image-comparison .before-image");
const sliderLine = document.querySelector(".image-comparison .slider-line");
const sliderIcon = document.querySelector(".image-comparison .slider-icon");

const vid_slider = document.querySelector(".vid-comparison .vslider");
const beforeVid = document.querySelector(".vid-comparison .before-vid");
const vid_sliderLine = document.querySelector(".vid-comparison .vslider-line");
const vid_sliderIcon = document.querySelector(".vid-comparison .vslider-icon");

const nerf_slider = document.querySelector(".nerf-comparison .nslider");
const beforeNerf = document.querySelector(".nerf-comparison .before-nerf");
const nerf_sliderLine = document.querySelector(".nerf-comparison .nslider-line");
const nerf_sliderIcon = document.querySelector(".nerf-comparison .nslider-icon");

slider.addEventListener("input", (e) => {
    let sliderValue = e.target.value + "%";

    beforeImage.style.width = sliderValue;
    sliderLine.style.left = sliderValue;
    sliderIcon.style.left = sliderValue;
});

vid_slider.addEventListener("input", (e) => {
    let sliderValue = e.target.value + "%";

    beforeVid.style.width = sliderValue;
    vid_sliderLine.style.left = sliderValue;
    vid_sliderIcon.style.left = sliderValue;
});

nerf_slider.addEventListener("input", (e) => {
    let sliderValue = e.target.value + "%";

    beforeNerf.style.width = sliderValue;
    nerf_sliderLine.style.left = sliderValue;
    nerf_sliderIcon.style.left = sliderValue;
});

function showImage(beforePath, stylePath, afterPath) {
    const before = document.getElementById("before-image");
    const content = document.getElementById("content");
    const style = document.getElementById("style");
    const after = document.getElementById("after-image");
    before.src = beforePath;
    content.src = beforePath;
    style.src = stylePath;
    after.src = afterPath;
}

function showVideo(beforePath, stylePath, afterPath) {
    const before = document.getElementById("before-vid");
    const content = document.getElementById("vid-content");
    const style = document.getElementById("vid-style");
    const after = document.getElementById("after-vid");
    before.src = beforePath;
    content.src = beforePath;
    style.src = stylePath;
    after.src = afterPath;
}

document.addEventListener('DOMContentLoaded', function() {
    
    const videoPairs = [
        [document.getElementById('before-vid'), document.getElementById('after-vid')],
        [document.getElementById('before-nerf'), document.getElementById('after-nerf')],
        // Add more pairs if needed
    ];

    videoPairs.forEach(pair => {
        const [videoA, videoB] = pair;

        videoA.addEventListener('timeupdate', function() {
            if (Math.abs(videoA.currentTime - videoB.currentTime) > 0.1) {
                videoB.currentTime = videoA.currentTime;
            }
        });

        videoB.addEventListener('timeupdate', function() {
            if (Math.abs(videoB.currentTime - videoA.currentTime) > 0.1) {
                videoA.currentTime = videoB.currentTime;
            }
        });

        videoA.addEventListener('play', function() {
            if (videoB.paused && !videoB.ended) {
                videoB.play();
            }
        });

        videoA.addEventListener('pause', function() {
            if (!videoB.paused && !videoB.ended) {
                videoB.pause();
            }
        });

        videoB.addEventListener('play', function() {
            if (videoA.paused && !videoA.ended) {
                videoA.play();
            }
        });

        videoB.addEventListener('pause', function() {
            if (!videoA.paused && !videoA.ended) {
                videoA.pause();
            }
        });
    });
});

$(document).ready(function() {
    var editor = CodeMirror.fromTextArea(document.getElementById("bibtex"), {
        lineNumbers: false,
        lineWrapping: true,
        readOnly:true
    });
    $(function () {
        $('[data-toggle="tooltip"]').tooltip()
    });
});