#define echoPin 2 
#define trigPin 3 


long duration; // variable for the duration of sound wave travel
int distance; // variable for the distance measurement

int count = 0;
int threshold = 130;
int Th = 400;
bool flag = true;

void setup() {
  pinMode(trigPin, OUTPUT); 
  pinMode(echoPin, INPUT); 
  Serial.begin(9600); 
  Serial.println("-----Footfall Counter-----"); 
}
void loop() {
  int d1,d2;
  d1 = read_distance();
  

if(distance < threshold){
  delay(Th);
  d2 = read_distance();
  if(d2>=d1)
   {
    count = count -1;
    flag = true;
   }
  else 
   {
    count = count +1;
    flag = true;
    delay(200);
   }
     delay(500);
  }
else
  {
  delay(500);
  }
if(count<0)
  {
   count = 0;
  }
  if(flag){
  Serial.println("No. of person in room :");
  Serial.println(count);
  }
  flag = false;
}

int read_distance(){

  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
 
  duration = pulseIn(echoPin, HIGH);
  
  distance = duration * 0.034 / 2; 
  
return distance;
}
