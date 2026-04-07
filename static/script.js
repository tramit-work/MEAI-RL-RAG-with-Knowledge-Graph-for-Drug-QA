async function askQuestion(){

let input = document.getElementById("question");
let question = input.value.trim();

if(question === "") return;

let chat = document.getElementById("chat");

/* USER MESSAGE */

let user = document.createElement("div");
user.className = "message user";
user.innerText = question;
chat.appendChild(user);

/* AI MESSAGE */

let ai = document.createElement("div");
ai.className = "message ai";
ai.innerText = "MeAI is analyzing medical knowledge...";
chat.appendChild(ai);

input.value="";
chat.scrollTop = chat.scrollHeight;

try{

let response = await fetch("/ask",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
question:question
})
});

let data = await response.json();

/* FORMAT ANSWER */

let formatted = data.answer
.replace("Answer:", "")
.trim();

ai.innerText = formatted;

}catch(error){

ai.innerText = "AI server error.";

}

chat.scrollTop = chat.scrollHeight;
}

function handleEnter(event){
if(event.key === "Enter"){
askQuestion();
}
}