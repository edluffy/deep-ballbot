
float target = 0.0;

void setup()
{
    Serial.begin(115200);
    delay(1000);
}

void loop()
{
    static String recieved_chars;
    
    while(Serial.available()){
        char inChar = (char) Serial.read();
        recieved_chars += inChar;
        if(inChar == '\n'){
            target = recieved_chars.toFloat();
            Serial.print("Target = ");
            Serial.println(target);
            recieved_chars = "";
        }
    }
}
