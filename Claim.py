import random
import matplotlib.pyplot as plt
#Loss_Ratio_Actual,Confidence_Level_Actual,Premium, Profit_Loss, rate,length = main(length,rate,confidence,loss)
                
class Report():
    # We want to record our reports on our simulations at different variables so we can do sensitivity analysis
    def __init__(self,loss_max,loss_actual, confidence_level_actual, confidence_level_min,premium,profit_loss,rate,length):
        self.loss_max = loss_max
        self.loss_actual = loss_actual
        self.confidence_level_actual = confidence_level_actual
        self.confidence_level_min = confidence_level_min
        self.premium = premium
        self.profit_loss = profit_loss
        self.risk_free_rate = rate
        self.period_length = length
    def Print(self):
         return print("Max Loss: %s%% Actual Loss: %s%%  Confidence: %s%%  Premium: $%s  P&L: $%s  risk-free rate %s%% over %s months" % (self.loss_max,self.loss_actual,self.confidence_level_actual,self.premium,self.profit_loss,self.risk_free_rate,self.period_length))

#Length_Of_Simulation,Risk_Free_Rate_Yearly,Amount_Of_Simulations,Premium,Loss_Ratio,ListOfClaims
class Simulation():
    def __init__(self,Length_Of_Simulation,Risk_Free_Rate_Yearly,Amount_Of_Simulations,Premium,Loss_Ratio):
        self.length = Length_Of_Simulation
        self.rate = Risk_Free_Rate_Yearly
        self.amount = Amount_Of_Simulations
        self.premium = Premium
        self.loss_ratio = Loss_Ratio
        self.data = [[0 for amount in range(self.length)] for length in range(self.amount)]
        self.account_list = [Account(self) for account in range(self.amount)]
        self.profit_rate = 0

        

    def Simulate(self):
        #Simulation of accumulated values
        Monthly_Equivalent = 12*((1+self.rate)**(1/12)-1)
        occurrence_probability = random.randint(0,100)
        Total_Premium_Paid = self.length * self.premium
        account = 0
        while account < self.amount:
            #simulating each account
            month =0
            while month < self.length:
                self.account_list[account].value[month] += self.premium #collect premium
                mu = random.randint(1000,19000)
                Claim_Check = Claim(mu,mu*random.uniform(0,2),random.randint(0,60)) #creating our possible claim
                if Claim_Check.Claim_Occurred(occurrence_probability) is True:  #checking to see if claim occurred
                    if Claim_Check.value < 0: #making sure we don't have negative claim values since we are using a normal distribution
                        Claim_Check.value = 0
                    self.account_list[account].claims.append(Claim_Check) #collect our claim on the account level
                    for claim in self.account_list[account].claims:
                        if claim.claim_closed is False:
                           claim.Claim_Development() #this is where we see how much is left on claims, if the claims need to be closed, and prepare to pay them out
                           self.account_list[account].value[month] -= claim.payment_amount #reduce our accumulated value by the payment amount
                self.account_list[account].value[month] *= (Monthly_Equivalent + 1) #compound by the equivalent risk free interest rate
                self.data[account][month] = self.account_list[account].value[month] #record the data and keep looping
                month+=1
            self.account_list[account].final_value = self.account_list[account].value[-1] #end of the account loop
            account+=1
        Positive_Count = 0
        final  = 0
        while final < len(self.account_list):
            if self.account_list[final].final_value > 0:
                Positive_Count +=1
            final +=1
        self.profit_rate = Positive_Count/len(self.account_list) #Amount that saw a profit in this simulation
        return self
        #end Simulate()

class Account():
    def __init__(self,simulation):
        self.value = [0 for value in range(simulation.length)]
        self.final_value = 0
        self.claims = []





class Claim():
    def __init__(self,mu,sigma,occurrence_probability):
        self.mu = mu
        self.sigma = sigma
        self.occurrence_probability = occurrence_probability
        self.value = round(random.normalvariate(self.mu,self.sigma))
        self.development_type = random.choice(["one","constant","balloon"])
        self.claim_closed = False
        self.claim_development_length = random.randint(2,12)
        self.claim_development = 0
        self.claim_value_remaining = self.value
        self.claim_paid = 0
        self.payment_amount = 0
    def Division_Amount(self):
        division_amount = 0
        index = 1
        while index <= self.claim_development_length:
            division_amount += index
            index+=1
        return division_amount
    def Claim_Occurred(self,occurrence_probability):
        if self.occurrence_probability >  occurrence_probability:
            return True
    def Claim_Development(self):
        if self.claim_value_remaining < 1:
            self.claim_closed = True

        if self.claim_closed is False:
            if self.development_type == "one":
                self.claim_closed = True
                self.claim_development_length = 0
                self.claim_value_remaining = 0
                self.claim_paid = self.value
                self.payment_amount = self.value
            elif self.development_type == "constant":
                Amount_To_Pay = self.value/self.claim_development_length
                if self.claim_development < self.claim_development_length:
                    self.claim_development += 1
                    self.claim_value_remaining -= Amount_To_Pay
                    self.claim_paid += Amount_To_Pay
                    self.payment_amount = Amount_To_Pay
                else:
                    self.claim_closed = True
            elif self.development_type == "balloon":
                division_amount = self.Division_Amount()
                    
                if self.claim_development < self.claim_development_length:
                    self.payment_amount = self.value/division_amount*self.claim_development
                    self.claim_paid+=self.payment_amount
                    self.claim_value_remaining -= self.payment_amount
                    self.claim_development+=1
                else:
                    self.claim_closed = True
            elif self.development_type == "decaying":
                division_amount = self.Division_Amount()
                if self.claim_development < self.claim_development_length:
                    self.payment_amount = self.value/division_amount*(self.claim_development_length-self.claim_development+1)
                    self.claim_paid+=self.payment_amount
                    self.claim_value_remaining -= self.payment_amount
                    self.claim_development+=1
                else:
                    self.claim_closed = True
        return self


def sens(a,a_list_parameters,b,rate,Loss_Ratio,Confidence_Level,Length_Of_Simulation,Premium):
    #Sensitivity analysis function
    # Do not call yet!
    list = [a,b]
    for x in list:
        if x == rate:
            x = rate
        elif x == Loss_Ratio:
            x = Loss_Ratio
        elif x == Confidence_Level:
            x = Confidence_Level
        elif x == Length_Of_Simulation:
            x = Length_Of_Simulation
    a_list_output = []
    b_list_output = []
    delta_b_over_delta_a = 0
    index = len(a_list_parameters)-1
    for a in a_list_parameters:
        Loss_Ratio_Actual,Confidence_Level_Actual,Premium, Profit_Loss, rate,length = main(Length_Of_Simulation,rate,Confidence_Level,Loss_Ratio)
        a_list_output.append(a)
        b_list_output.append(b)
    while index > 0:
        partial_a = a_list_output[index]-a_list_output[index-1]
        partial_b = b_list_output[index]-b_list_output[index-1]
        index-=1
        delta_b_over_delta_a += partial_b/partial_a
    return round(float(delta_b_over_delta_a),5)
                





def comma(value):
    #prints our values nicely :-)
    return '{:,}'.format(round(value))

def MM(value):
    #convert long integer into value MM
    if value >= 1000000:
        value/=1000000
    return str(comma(value))+" MM"


def convert(value):
    #Normalize against dinguses that type 30 or .3
    if 1<value<=100:
        value/=100
        return value
    elif value < 0 :
        return 0
    else:
        return value
'''
def Simulate(Length_Of_Simulation,Risk_Free_Rate_Yearly,Amount_Of_Simulations,Premium,Loss_Ratio,ListOfClaims):
    #Simulation of accumulated values
    Monthly_Equivalent = 12*((1+Risk_Free_Rate_Yearly)**(1/12)-1)
    occurrence_probability = random.randint(0,100)
    Total_Premium_Paid = Length_Of_Simulation * Premium
    #print(Total_Premium_Paid)
    Data = [[0 for month in range(Length_Of_Simulation)] for account in range(Amount_Of_Simulations)] #initializse the data we are going to collect
    Final_Values = [0 for account in range(Amount_Of_Simulations)] #The Final Accumulated values for our streams
    for account in range(Amount_Of_Simulations):
        #simulating each account
        Value = 0
        for month in range(Length_Of_Simulation):
            Value += Premium #collect premium
            mu = random.randint(1000,19000)
            Claim_Check = Claim(mu,mu*random.uniform(0,2),random.randint(0,60)) #creating our possible claim
            if Claim_Check.Claim_Occurred(occurrence_probability) is True:  #checking to see if claim occurred
                #print("claim!")
                if Claim_Check.value < 0: #making sure we don't have negative claim values since we are using a normal distribution
                    Claim_Check.value = 0
                ListOfClaims[account].append(Claim_Check) #collect our claim on the account level
                for claim in ListOfClaims[account]:
                    if claim.claim_closed is False:
                       claim = claim.Claim_Development() #this is where we see how much is left on claims, if the claims need to be closed, and prepare to pay them out
                       Value -= claim.payment_amount #reduce our accumulated value by the payment amount
            Value *= (Monthly_Equivalent + 1) #compound by the equivalent risk free interest rate
            Data[account][month] = Value #record the data and keep looping
        Final_Values[account] = Value #end of the account loop
        #print(Data)
    Positive_Count = 0
    for account in Final_Values:
        if account > 0:
            Positive_Count +=1
    Profit_Rate = Positive_Count/len(Final_Values) #Amount that saw a profit in this simulation
    return Profit_Rate,Premium,Data
    #end Simulate()
'''


def main(Length_Of_Simulation,Risk_Free_Rate_Yearly,Confidence_Level,Loss_Ratio):
    Amount_Of_Simulations = 1 #amount of policies we are generating
    Profit_Rate = 0
    Profit_Count_Ratio = 0
    Test_Run_Amount = 200 #the test sample to test against once we find our loss ratio we are looking for
    Premium = 0
    while Profit_Count_Ratio < Confidence_Level: # we are going to keep increasing premiums until we hit our loss ratio and confidence level
        Simulation_Variable = Simulation(Length_Of_Simulation,Risk_Free_Rate_Yearly,Amount_Of_Simulations,Premium,Loss_Ratio)
        Simulation_Variable.Simulate()
        Profit_Count = 0
        Average_Profit_Rate = 0
        Plot_Data = []
        if Simulation_Variable.profit_rate >= 1-Loss_Ratio:
            for simulation in range(Test_Run_Amount):
                Simulation_Variable = Simulation(Length_Of_Simulation,Risk_Free_Rate_Yearly,Amount_Of_Simulations,Premium,Loss_Ratio)
                Simulation_Variable.Simulate()
                Plot_Data.append(Simulation_Variable.data)
                Average_Profit_Rate += Simulation_Variable.profit_rate
                if Simulation_Variable.profit_rate >= 1-Loss_Ratio:
                    Profit_Count +=1
            Average_Profit_Rate /= Test_Run_Amount
            Profit_Test_Ratio = Profit_Count/Test_Run_Amount #amount of simulations that passed the loss ratio in our test run
            if Profit_Test_Ratio >= Confidence_Level and Average_Profit_Rate >= 1-Loss_Ratio:
                break
        #print("increase premiums to %s" % (Simulation_Variable.premium+100))
        Premium += 100
    total_gains = 0
    total_losses = 0
    gain_count = 0
    loss_count = 0
    for simulation in range(len(Plot_Data)):
        for test_sub_simulation in range(len(Plot_Data[simulation])):
            plt.plot(Plot_Data[simulation][test_sub_simulation])
            plt.ylabel('Account Value')
            plt.xlabel('Months')
            if Plot_Data[simulation][test_sub_simulation][-1] > 0:
                gain_count +=1
                total_gains += Plot_Data[simulation][test_sub_simulation][-1]
            else:
                loss_count +=1
                total_losses += Plot_Data[simulation][test_sub_simulation][-1]
    if loss_count == 0:
        loss_count = 1
    if gain_count == 0:
        gain_count = 1
    avg_loss = total_losses/loss_count
    avg_gain = total_gains/gain_count
    final_loss_ratio = loss_count/(loss_count+gain_count)
    total_premium = Length_Of_Simulation*Amount_Of_Simulations*Test_Run_Amount*Premium
    total_claims = 0
    account = 0
    total_claim_paid = 0
    total_remaining = 0
    open_claims = 0
    total_value = 0
    while account < (len(Simulation_Variable.account_list)):
        for claim in Simulation_Variable.account_list[account].claims:
            total_value +=claim.value
            total_claims +=1
            total_claim_paid += claim.claim_paid
            total_remaining += claim.claim_value_remaining
            if claim.claim_closed is False:
                open_claims += 1
        account+=1
    P_L = total_gains - total_losses
    #print("With premium of %s we will achieve a loss ratio of %s%% or better over a period of %s months with %s%% confidence" % (Premium,round(final_loss_ratio*100),Length_Of_Simulation,(Profit_Test_Ratio)*100))
    #print("Actual loss amount: $ %s Actual profit amount: $ %s Total policy P&L: $ %s and collected $%s in premium" % (comma(round(total_losses)),comma(round(total_gains)),comma(round(total_gains+total_losses)),comma(total_premium)))
    #print("Average loss amount: $ %s Average profit amount: $ %s Average Policy P&L: $ %s" %(comma(round(avg_loss)),comma(round(avg_gain)),comma(round((avg_gain+avg_loss)/(gain_count+loss_count)))))
    #print("There were %s claims, $%s paid out of value %s, with $%s remaining and %s open claims remaining" % (total_claims,comma(total_claim_paid),comma(total_value),comma(total_remaining),open_claims))
    #print("On average, there were %s claims per period per account" % round(total_claims/(Length_Of_Simulation*Test_Run_Amount*Amount_Of_Simulations),2))
    #plt.show()
    return final_loss_ratio,Profit_Test_Ratio, Premium, P_L, Risk_Free_Rate_Yearly,Length_Of_Simulation
Length_Of_Simulation = [12,24]
Loss_Ratio = [0.35,.1]
Confidence_Level = [.85,.9]
Risk_Free_Rate_Yearly = [.035,0.075]
List_Of_Reports = []
Total_Reports = len(Length_Of_Simulation)*len(Loss_Ratio)*len(Risk_Free_Rate_Yearly)*len(Confidence_Level)
for loss in Loss_Ratio:
    for confidence in Confidence_Level:        
        for rate in Risk_Free_Rate_Yearly:
            for length in Length_Of_Simulation:
                Loss_Ratio_Actual,Confidence_Level_Actual,Premium, Profit_Loss, rate,length = main(length,rate,confidence,float(loss))
                List_Of_Reports.append(Report(loss,Loss_Ratio_Actual,Confidence_Level_Actual,confidence,Premium,Profit_Loss,rate,length))
for report in List_Of_Reports: 
    break               
    test = sens(report.risk_free_rate,Risk_Free_Rate_Yearly,report.premium,report.risk_free_rate,report.loss_max,report.confidence_level_min,report.period_length,report.premium)
    print(test)
    
                #print("Appended %s out of %s!" % (len(List_Of_Reports),Total_Reports))
for report in List_Of_Reports:
    if report.premium < 1000:
        report.premium = str(comma(report.premium))+"  "
    else:
        report.premium = comma(report.premium)
    if report.confidence_level_actual*100 < 10:
        report.confidence_level_actual = str(comma(report.confidence_level_actual*100))+" "
    else:
        report.confidence_level_actual = comma(report.confidence_level_actual*100)
    if report.profit_loss < 10000000:
        report.profit_loss = comma(report.profit_loss)+" "
    else:
        report.profit_loss = comma(report.profit_loss)
    if report.loss_actual < .1:
        report.loss_actual = str(comma(report.loss_actual*100))+" "
    else:
        report.loss_actual = comma(report.loss_actual*100)
    report.loss_max = comma(float(report.loss_max)*100)
    report.risk_free_rate = round(report.risk_free_rate*100.0,2)
    
    report.Print()
                   