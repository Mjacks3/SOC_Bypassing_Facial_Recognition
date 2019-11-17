//Add Event listeners
/*
document.getElementById("initialize_training_button").addEventListener("click", function() {send_train_request();});


function send_train_request(){
    training_name =  document.getElementById("training_name").value;
    console.log("Training Name " + training_name);

    training_data =  document.getElementById("training_data").value;
    console.log("Training Data " + training_data);


    console.log("Request Received");


    var form_data = new FormData(document.getElementById("uploadform").elements[0].value);

    console.log(form_data);

    data = {
        "training_name":training_name
    }

    jQuery.get('/acct_creation', data, function(rtn_data) {

        console.log("Acct_creation Response Recieved");
        //console.log(rtn_data);
        
    })
    

    /*
    jQuery.post('/edgelist', JSON.stringify(data), function(data) {
        console.log("Response Recieved");
        console.log(data);
    }).fail(function(xhr, status, error) {
        // TODO - show this to user in UI. 
        console.log(xhr, status, error);
    }).always(function() {
        hide_spinner();
    });
    
};
*/

$(function() {
    $('#initialize_training_button').click(function() {
        event.preventDefault();
        var form_data = new FormData($('#uploadform')[0]);
        $.ajax({
            type: 'POST',
            url: '/acct_creation',
            data: form_data,
            contentType: false,
            processData: false,
            dataType: 'json'
        }).done(function(data, textStatus, jqXHR){
            console.log(data);
            console.log(textStatus);
            console.log(jqXHR);
            console.log('Success!');
            $("#resultFilename").text(data['name']);
            $("#resultFilesize").text(data['size']);
        }).fail(function(data){
            console.log('error!');
        });
    });
}); 