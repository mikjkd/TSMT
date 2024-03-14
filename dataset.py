import sys

sys.path.insert(1, '../../../')

from generate_dataset import *
import pickle as pkl


def generate_ds(seq_len_x, seq_len_y_par, data_path, base_path, columns, oheFiles, start_date=None, end_date=None):
    # frame contiene le informazioni passate e viene processato dalla rete per creare delle predizioni
    # info frame contiene le informazioni che la rete sfrutta per migliorare le predizioni

    # merge dataset
    df = merge_datasets(orders_path=data_path + 'annual_timeshifts.csv', orders_columns=columns,
                        access_path=data_path + 'annual_access_daily_full.csv',
                        weather_path=None, partite_path=data_path + 'partite.csv',
                        events_path=data_path + 'eventi.csv')
    if start_date:
        df = df[df['date'] >= start_date]

    if end_date:
        df = df[df['date'] <= end_date]

    # creazione info frame
    info_frame = df.copy()

    print(info_frame)

    # processo frame ed info_frame
    frame = process_ds(df, refill_zero=False, scaler_temp=None, addWeather = False, oheFiles=oheFiles)
    frame =frame.fillna(0)
    info_frame = preprocess_ds(info_frame, refill_zero=False, dropColumns=False)

    print(frame.columns)
    print(info_frame.columns)

    # scalo le features e rimuovo quelle inutili
    columns_to_scale = ['orders_completed', 'subtotal', 'shipping_cost', 'coupons_count', 'coupons_value',
                        'bookings_taken', 'bookings_auth', 'partners_with_orders', 'count']
    frame = scale_df(frame, columns_to_scale=columns_to_scale, scaler_path='../scalers/',scalerType='standard')
    seq_len = seq_len_x
    seq_len_y = seq_len_y_par  # 11 giorni
    frame_drop = frame.drop(['date', 'timeshift', 'city', 'month', 'is_cap','is_coupon'],
                            axis=1)
    info_frame_drop = info_frame.drop(['date', 'timeshift', 'city', 'orders_completed', 'subtotal',
                                       'shipping_cost', 'coupons_count', 'coupons_value', 'bookings_taken',
                                       'bookings_auth', 'partners_with_orders', 'count', 'month', 'is_cap','is_coupon'], axis=1)

    print(frame_drop.columns)
    print(info_frame_drop.columns)
    # creo le sequenze per la rete
    X, _ = split_sequence(frame_drop.values.astype('float64'), seq_len, seq_len_y)

    X1, _ = split_sequence(frame_drop.values.astype('float64'), seq_len + seq_len_y, 0)
    Y = np.empty((X1.shape[0], seq_len, seq_len_y))

    for step_ahead in range(1, seq_len_y + 1):
        Y[:, :, step_ahead - 1] = X1[:, step_ahead:step_ahead + seq_len, 0]

    X_info, _ = split_sequence(info_frame_drop.values.astype('float64'), seq_len + seq_len_y, 0)
    X_i = np.empty((X_info.shape[0], seq_len, seq_len_y, X_info.shape[2]))
    for step_ahead in range(1, seq_len_y + 1):
        X_i[:, :, step_ahead - 1] = X_info[:, step_ahead:step_ahead + seq_len, :]

    print(X.shape, X_i.shape, Y.shape)

    # pagino il dataset e lo salvo nell'apposita cartella
    filenames = []

    for idx, x in enumerate(X):
        fnx = f'X_data_base_11d_{idx}'
        fnxi = f'X_i_data_base_11d_{idx}'
        fny = f'Y_data_base_11d_{idx}'
        filenames.append([fnx, fnxi, fny])
        with open(base_path + fnx, 'wb') as output:
            pkl.dump(x, output)
        with open(base_path + fnxi, 'wb') as output:
            pkl.dump(X_i[idx], output)
        with open(base_path + fny, 'wb') as output:
            pkl.dump(Y[idx], output)
    np.save(f'{base_path}/filenames_base_11d.npy', filenames)


if __name__ == '__main__':
    data_path = '../../../data/'
    base_path = 'ds_paginato/'
    seq_len_x = 120
    seq_len_y_par = 44
    columns = ['date', 'timeshift', 'city',
               'orders_completed', 'subtotal', 'shipping_cost', 'coupons_count',
               'coupons_value', 'bookings_taken', 'bookings_auth',
               'partners_with_orders']
    oheFiles = {
        'month': '../encoders/oheMonth',
        'year': '../encoders/oheYears',
        'ts': '../encoders/oheTs',
        'micro': '../encoders/oheMicro'
    }
    generate_ds(seq_len_x, seq_len_y_par, data_path, base_path, columns, oheFiles, start_date=None, end_date=None)