library(tidyverse)
library(lubridate)
library(zoo)
library(forecast)
library(keras)
library(tensorflow)

# Convert file encoding before reading
file_path <- "C:/Users/mcken/OneDrive/Desktop/underwater_temperature.csv"
converted_file <- iconv(readLines(file_path), from = "latin1", to = "UTF-8")

# Write the converted content to a new file
writeLines(converted_file, "C:/Users/mcken/OneDrive/Desktop/underwater_temperature_converted.csv")

# Try reading the newly saved file
df <- read_csv("C:/Users/mcken/OneDrive/Desktop/underwater_temperature_converted.csv")

# View the first few rows of the dataset
head(df)

# Convert 'Date' to Date type
df$Date <- as.Date(df$Date)

# Sort the data by Date
df <- df %>% arrange(Date)

# Fill missing values using linear interpolation
df$Temperature <- na.approx(df$`Temp (°C)`)

# Plot the ocean temperature over time
ggplot(df, aes(x = Date, y = Temperature)) +
  geom_line() +
  labs(title = "Ocean Temperature Over Time", x = "Date", y = "Temperature (°C)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 2. ARIMA Model Implementation
# Fit ARIMA model (use auto.arima to automatically choose the best model)
fit_arima <- auto.arima(df$Temperature)

# Forecast the next 30 days
forecast_arima <- forecast(fit_arima, h = 30)

# Plot ARIMA forecast
autoplot(forecast_arima) +
  labs(title = "ARIMA Model Forecast for Ocean Temperature", x = "Date", y = "Temperature (°C)")

# Model Evaluation for ARIMA
mae_arima <- mean(abs(forecast_arima$residuals))
rmse_arima <- sqrt(mean(forecast_arima$residuals^2))

cat("ARIMA Model MAE: ", mae_arima, " RMSE: ", rmse_arima, "\n")

# 3. LSTM Model Implementation (Using Keras)
# Prepare the data for LSTM
temperature_values <- df$Temperature
temperature_scaled <- scale(temperature_values)

# Prepare training data (Use last 30 days to predict the next temperature)
X <- matrix(NA, nrow = length(temperature_scaled) - 30, ncol = 30)
y <- matrix(NA, nrow = length(temperature_scaled) - 30, ncol = 1)

for (i in 31:length(temperature_scaled)) {
  X[i - 30, ] <- temperature_scaled[(i - 30):(i - 1)]
  y[i - 30] <- temperature_scaled[i]
}

# Reshape data for LSTM (3D array)
X <- array(X, dim = c(nrow(X), ncol(X), 1))

# Build the LSTM model
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(30, 1)) %>%
  layer_lstm(units = 50) %>%
  layer_dense(units = 1)

model %>% compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit the LSTM model
model %>% fit(X, y, epochs = 10, batch_size = 32)

# Predict the next 30 days
predicted_temp <- model %>% predict(X[(nrow(X)-30+1):nrow(X), , , drop = FALSE])

# Inverse transform the predictions
predicted_temp <- scale(predicted_temp, center = attr(temperature_scaled, "scaled:center"), scale = attr(temperature_scaled, "scaled:scale"))

# Get the dates for the last 30 days in the dataset
forecast_dates <- df$Date[(nrow(df) - 29):nrow(df)]

# Plot observed data and predicted data together
plot(forecast_dates, df$Temperature[(nrow(df) - 29):nrow(df)], 
     type = "l", col = "blue", xlab = "Date", ylab = "Temperature (°C)", 
     main = "LSTM Model Prediction for Ocean Temperature")

# Add predicted values to the plot
lines(forecast_dates, predicted_temp, col = "red", lty = 2)

# Add a legend to the plot
legend("topright", legend = c("Observed", "Predicted"), col = c("blue", "red"), lty = 1:2)

# 4. Plotting Temperature Change Distribution
df$TemperatureChange <- c(NA, diff(df$Temperature))
df_clean <- df %>% filter(!is.na(TemperatureChange))
ggplot(df_clean, aes(x = TemperatureChange)) +
  geom_histogram(binwidth = 0.1, fill = "pink", color = "purple", alpha = 0.7) +
  labs(title = "Distribution of Daily Temperature Changes", x = "Temperature Change (°C)", y = "Frequency")
