$(function() {
    $('#initialize_training_button').click(function() {
        event.preventDefault();
        training_name = document.getElementById("training_name").value;

        document.getElementById("training_data").setAttribute("name",training_name);
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
           $.ajax({
                type: 'POST',
                url: '/train',
                data: JSON.stringify({"name":training_name}),
                contentType: false,
                processData: false,
                dataType: 'json'
            })

        }).fail(function(data){
            console.log('error!');
        });});}); 

