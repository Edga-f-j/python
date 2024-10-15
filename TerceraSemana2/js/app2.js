


const inputemail = document.getElementById("email");
const inputclave = document.getElementById("clave");
const miElemento = document.getElementById("miElemento")
const myForm = document.querySelector("form");

myForm.addEventListener("click", function(event){
    event.preventDefault();
});

function validarform(){

    const expresion = new RegExp('^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$');
    if(expresion.test(inputemail.value)){
        console.log("Si ");
      
    }
    else{
        alert("error en el correo");
        return;
    }

    if(inputclave.value === ""){
        alert("debe ingresar una clave");
    }
    
    else {
        if(inputclave.value == "miclave"){
            console.log(miElemento)
            alert("ha ingresado correctamente");
            miElemento.innerHTML = "Bienvenido "+ inputemail.value;
        }else{
            alert("la clave esta mal");
        }   
    }
}








// function escala(x){
//     for(let i=1;i<=20;i++){
//                 x*=i;
//                 console.log(i+". = "+x);

//                 if(i==20){
//                     for(let j=20;j>1;j--){
//                         x/=j;
//                         console.log((j-1)+". = "+x);
//                         }      
//                      }
//     }
// }

// let y=10
// escala(y);

// let lista =[];


// for(let carro of carros){
//     if(carro.mpg>30){
//         lista.push(carro.marca)
//     }

// }
// console.log(lista)





