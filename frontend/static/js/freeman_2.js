document.getElementById("manual_radio").addEventListener("change", function()
{
    document.getElementById("uploadform3").style.display = 'none';
    document.getElementById("uploadform2").style.display = 'block';

});

document.getElementById("auto_radio").addEventListener("change", function()
{
    document.getElementById("uploadform2").style.display = 'none';
    document.getElementById("uploadform3").style.display = 'block';

});



$(function() {
    $('#initialize_test_button').click(function() {
        event.preventDefault();
        test_name = document.getElementById("test_name").value;

        document.getElementById("test_data").setAttribute("name",test_name)
        var form_data = new FormData($('#uploadform2')[0]);
        $.ajax({
            type: 'POST',
            url: '/acct_test',
            data: form_data,
            contentType: false,
            processData: false,
            dataType: 'json'
        }).done(function(data, textStatus, jqXHR){
            console.log(data);
            console.log(textStatus);
            console.log(jqXHR);
            console.log('Success!');
        }).fail(function(data){
            console.log('error!');
        });
    });
}); 


var width = 320;
var height = 0;
var streaming = false;
var latest_data_url;

navigator.mediaDevices.getUserMedia({video: true, audio: false})
    .then(function (stream) {video.srcObject = stream; video.play();})
    .catch(function (err) {console.log("An error occured! " + err);});

video.addEventListener('canplay', function (ev) {
if (!streaming) {
    height = video.videoHeight / (video.videoWidth / width);
    video.setAttribute('width', width);
    video.setAttribute('height', height);
    canvas.setAttribute('width', width);
    canvas.setAttribute('height', height);

    streaming = true;
}
}, false);

startbutton.addEventListener('click', function (ev) {
takepicture();
ev.preventDefault();
}, false);

clearphoto();

function clearphoto() {
var context = canvas.getContext('2d');
context.fillStyle = "#AAA";
context.fillRect(0, 0, canvas.width, canvas.height);
}

function takepicture() 
{
var context = canvas.getContext('2d');
if (width && height) 
{
    canvas.width = width;
    canvas.height = height;
    context.drawImage(video, 0, 0, width, height);
    var dataURL = canvas.toDataURL("image/jpeg");
    latest_data_url = dataURL;

    if (dataURL && dataURL != "data:,") 
    {
        //nothng
    } 
    else 
    {
        alert("Image not available");
    }
}   
else 
{
    clearphoto();
}
}

$(function() {

$('#initialize_test_button_v2').click(function() {
    event.preventDefault();
    test_name = document.getElementById("test_name").value;

    $.ajax({
        type: 'POST',
        url: '/acct_test_v2',
        data: {"image": latest_data_url,
                "name": test_name}
    }).done(function(data, textStatus, jqXHR){
        console.log(data);
        console.log(textStatus);
        console.log(jqXHR);
        console.log('Success!');
    }).fail(function(data){
        console.log('error!');
    });
});
}); 




