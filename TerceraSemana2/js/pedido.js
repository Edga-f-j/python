

const inputqueso = document.getElementById("queso");
const inputpan = document.getElementById("pan");
const inputcarne = document.getElementById("carne");
const inputlechuga = document.getElementById("lechuga");
const miElemento = document.getElementById("miElemento");
const myForm = document.querySelector("form");



myForm.addEventListener("click", function(event){
    event.preventDefault();
});

function validarform(){
    if(inputqueso.value ===""){
        alert("!!ERROR!! debe ingresar un dato");
    }
    if(inputcarne.value ===""){
        alert("!!ERROR!! debe ingresar un dato");
    }
    if(inputpan.value ===""){
        alert("!!ERROR!! debe ingresar un dato");
    }
    if(inputlechuga.value ===""){
        alert("!!ERROR!! debe ingresar un dato");
    }
    else{
        console.log(miElemento)
        alert("Su pedido esta en marcha");
        miElemento.innerHTML = "Su pedido consta de: carne con "+ inputcarne.value;
    }

}

document.getElementById('pedidoForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Evita el envío del formulario

    // Obtener los valores de los campos del formulario
    const nombre = document.getElementById('nombre').value;
    const email = document.getElementById('email').value;
    const producto = document.getElementById('producto').value;
    const cantidad = document.getElementById('cantidad').value;

    // Mostrar los datos en la consola (puedes enviarlos a un servidor aquí)
    console.log(`Nombre: ${nombre}`);
    console.log(`Email: ${email}`);
    console.log(`Producto: ${producto}`);
    console.log(`Cantidad: ${cantidad}`);

    // Aquí puedes agregar el código para enviar los datos a un servidor
    alert('Pedido enviado con éxito');
});
