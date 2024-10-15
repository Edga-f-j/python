



// function edad(x){

//     if (x<18){
//         return console.log("Eres menor de edad pana");
//     } else if(x>=18 && x<65){
//         return console.log("Eres adulto");
//     }else{
//         return console.log("Tas viejo");
//     }
// }

// const x =17;
// edad(x);

// otra forma de escribir con el if


// ========================================for con una funcion
// function edad(x){

//     for(i=0;i<2;i++){
    
//         if(i==0){
//         let mensaje = (x < 18) ?"menor.":"mayor";
//         console.log(mensaje);
//         }else if(i==1){
//             x=x+30;
//         let mensaje = (x >= 18 && x<60)? "Es mayor de edad": "Es viejo";
//         console.log(mensaje);
//         }else{
//             x=x+30;
//         let mensaje = (x<60 )? "En el 2084 seras viejo": "Es joven"
//         console.log(mensaje);
//         }
//     }
//     }
    


// let day = 3;
// switch (day){
//     case 1:
//         console.log("Lunes");
//         break;
//     case 2:
//         console.log("Martes");
//         break;
//     case 3:
//         console.log("Miercoles");
//         break;
//     default:
//         console.log("Invalido");
// }


 for(let carro of carros){
     console.log(carro.marca);
}
