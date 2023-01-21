from sklearn.preprocessing import MinMaxScaler
import torch
from dataset import SP500Dataset
from torch.utils.data import DataLoader

def predict(model, symbol, N):

    predict_ds =  SP500Dataset('sp500_test.csv', mode = 'predict')
    valid_symbols = predict_ds.symbols
    try:
        symbol_index = valid_symbols.index(symbol)
    except:
        return None
    
    predict_loader = DataLoader(predict_ds, batch_size = len(valid_symbols))
    old_periods, _ =  next(iter(predict_loader))
    last_year_prices = old_periods[:, -1]
    all_predicts = []
    for i in range(12):
    
        next_period_predict = model.forward(old_periods.unsqueeze(-1)).unsqueeze(-1)

        old_periods = torch.cat((old_periods[:, 1:].to(next_period_predict.device), next_period_predict), axis = 1)
        all_predicts.append(next_period_predict)
    

    all_predicts = torch.cat(all_predicts, dim = 1)

    all_final_predicts = all_predicts[:, -1].unsqueeze(-1)

    inverse_scaler = MinMaxScaler()
    inverse_scaler.min_, inverse_scaler.scale_ = predict_ds.scaler.min_[0],predict_ds.scaler.scale_[0]

    last_year_prices = torch.Tensor(inverse_scaler.inverse_transform(last_year_prices.unsqueeze(-1))).squeeze()
    all_final_predicts = torch.Tensor(inverse_scaler.inverse_transform(all_final_predicts.cpu().detach())).squeeze()
    all_predicts = torch.Tensor(inverse_scaler.inverse_transform(all_predicts.cpu().detach())).squeeze()


    growth_rates = torch.div(all_final_predicts.squeeze(), last_year_prices)
    
    sorted_growth_rates, sorted_indexs = torch.sort(growth_rates, descending = True)

    topN_growth_rates = sorted_growth_rates[:N].tolist()

    topN_predict_prices = torch.index_select(all_predicts, 0, sorted_indexs)[:N].tolist()
    sorted_indexs = sorted_indexs.tolist()

    topN_symbols = []
    for idx in sorted_indexs[:N]:
        topN_symbols.append(valid_symbols[idx])
    symbol_predict_prices = all_predicts[symbol_index].tolist()
    symbol_predict_prices.append(growth_rates[symbol_index].item())
    topN_data = {
        
    }
    for i, symbol in enumerate(topN_symbols):
        topN_data[symbol] = topN_predict_prices[i]
        topN_data[symbol] .append(topN_growth_rates[i])
    return symbol_predict_prices, topN_data
    

