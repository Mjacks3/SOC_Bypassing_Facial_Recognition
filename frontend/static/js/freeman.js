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
        }).fail(function(data){
            console.log('error!');
        });});}); 


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