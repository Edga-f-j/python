
function agregar(valor){
    document.getElementById('Pantalla').value += valor;
}

function borrar(){
    document.getElementById('Pantalla').value ='';
}


function calcular(){
    const valor = document.getElementById('Pantalla').value;
    const resultado = eval(valor)
    document.getElementById('Pantalla').value = resultado;
}

