package forecast

import (
	"context"

	"github.com/rocketlaunchr/dataframe-go"
)

// Model is an interface to group trained models of Different
// Algorithms in the Forecast package under similar generic standard
type Model interface {
	// Predict method is used to run future predictions for Ses
	// Using Bootstrapping method
	Predict(ctx context.Context, m int) (*dataframe.SeriesFloat64, error)
	// Summary function is used to Print out Data Summary
	// From the Trained Model
	Summary()
	// Optimize method tunes the model result and tries to reduce
	// Accuracy Errors To the mininum
	Optimize() (*SesModel, error)
}
